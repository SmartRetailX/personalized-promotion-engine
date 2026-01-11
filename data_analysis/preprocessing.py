"""
Data preprocessing and feature engineering for ML models
Creates features from transaction data for personalized promotions

FIXES APPLIED:
1. Added product_category_encoded feature (FIX #1)
2. Time-aware training data with no future leakage
3. Time-based train/validation/test split (FIX #2)
4. create_customer_product_matrix with time cutoff (FIX #5)
5. Log-scaled interaction scores (FIX #6)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import os


class DataPreprocessor:
    """Preprocess raw transaction data for ML models"""
    
    def __init__(self, data_dir='f:\\.1 Research\\Personalized Promotion Engine\\data\\raw'):
        self.data_dir = data_dir
        self.customers = None
        self.products = None
        self.transactions = None
        self.promotions = None
        self.category_encoder = LabelEncoder()  # FIX #1: Category encoder
        
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        
        self.customers = pd.read_csv(os.path.join(self.data_dir, 'Customers.csv'))
        self.products = pd.read_csv(os.path.join(self.data_dir, 'Products.csv'))
        self.transactions = pd.read_csv(os.path.join(self.data_dir, 'Transactions.csv'))
        self.promotions = pd.read_csv(os.path.join(self.data_dir, 'Promotions.csv'))
        
        # Convert dates
        self.transactions['TransactionDate'] = pd.to_datetime(self.transactions['TransactionDate'])
        self.customers['RegistrationDate'] = pd.to_datetime(self.customers['RegistrationDate'])
        
        # FIX #1: Fit category encoder
        self.category_encoder.fit(self.products['Category'].unique())
        
        print(f"  Loaded {len(self.customers)} customers")
        print(f"  Loaded {len(self.products)} products")
        print(f"  Loaded {len(self.transactions)} transactions")
        print(f"  Loaded {len(self.promotions)} promotions")
        print(f"  Categories: {list(self.category_encoder.classes_)}")
        
    def create_customer_features(self, transactions=None, reference_date=None):
        """Create customer-level features"""
        print("\nCreating customer features...")
        
        if transactions is None:
            transactions = self.transactions
        if reference_date is None:
            reference_date = transactions['TransactionDate'].max()
        
        customer_features = transactions.groupby('CustomerID').agg({
            'TransactionID': 'count',
            'TotalAmount': ['sum', 'mean', 'std'],
            'TransactionDate': lambda x: (reference_date - x.max()).days,
            'Quantity': ['sum', 'mean'],
            'DiscountedAmount': ['sum', 'mean']
        }).reset_index()
        
        customer_features.columns = ['CustomerID', 'purchase_frequency', 
                                     'total_spent', 'avg_transaction_value', 'std_transaction_value',
                                     'recency_days', 'total_items_purchased', 'avg_items_per_transaction',
                                     'total_discounts_received', 'avg_discount_per_transaction']
        
        customer_features['std_transaction_value'] = customer_features['std_transaction_value'].fillna(0)
        
        promo_transactions = transactions[transactions['PromotionID'] != 'None']
        promo_response = promo_transactions.groupby('CustomerID').size().reset_index(name='promo_purchases')
        customer_features = customer_features.merge(promo_response, on='CustomerID', how='left')
        customer_features['promo_purchases'] = customer_features['promo_purchases'].fillna(0)
        customer_features['promo_response_rate'] = customer_features['promo_purchases'] / customer_features['purchase_frequency']
        
        customer_features = customer_features.merge(
            self.customers[['CustomerID', 'Name', 'Age', 'Gender', 'Location', 'CustomerSegment']], 
            on='CustomerID'
        )
        
        return customer_features
    
    def create_product_features(self, transactions=None):
        """Create product-level features"""
        print("Creating product features...")
        
        if transactions is None:
            transactions = self.transactions
        
        product_features = transactions.groupby('ProductID').agg({
            'TransactionID': 'count',
            'CustomerID': 'nunique',
            'TotalAmount': ['sum', 'mean'],
            'Quantity': 'sum'
        }).reset_index()
        
        product_features.columns = ['ProductID', 'total_purchases', 'unique_customers',
                                    'total_revenue', 'avg_revenue_per_purchase', 'total_quantity_sold']
        
        product_features = product_features.merge(
            self.products[['ProductID', 'Category', 'Brand', 'Price', 'PurchaseFrequency']],
            on='ProductID'
        )
        
        # FIX #4: Removed product_total_purchases from popularity (correlated)
        # Using only unique_customers for popularity
        product_features['popularity_score'] = (
            product_features['unique_customers'] / product_features['unique_customers'].max()
        )
        
        return product_features
    
    def create_customer_product_matrix(self, end_date=None):
        """
        Create customer-product interaction matrix for collaborative filtering
        
        FIX #5: Added end_date parameter to prevent future leakage
        FIX #6: Using log-scaled interaction scores
        """
        print("Creating customer-product interaction matrix...")
        
        # FIX #5: Filter by end_date to prevent future leakage
        if end_date is not None:
            transactions = self.transactions[self.transactions['TransactionDate'] < end_date]
            print(f"  Using transactions up to {end_date.date()}")
        else:
            transactions = self.transactions
        
        interactions = transactions.groupby(['CustomerID', 'ProductID']).agg({
            'Quantity': 'sum',
            'TotalAmount': 'sum',
            'TransactionID': 'count'
        }).reset_index()
        
        interactions.columns = ['CustomerID', 'ProductID', 'total_quantity', 
                               'total_spent', 'purchase_count']
        
        # FIX #6: Log-scaled interaction score (reduces heavy buyer dominance)
        interactions['interaction_score'] = np.log1p(interactions['purchase_count'])
        
        print(f"  Created {len(interactions)} interaction records")
        
        return interactions
    
    def create_category_preferences(self, transactions=None):
        """Calculate customer preferences for each product category"""
        print("Creating category preference features...")
        
        if transactions is None:
            transactions = self.transactions
        
        trans_with_category = transactions.merge(
            self.products[['ProductID', 'Category']], 
            on='ProductID'
        )
        
        category_prefs = trans_with_category.groupby(['CustomerID', 'Category']).agg({
            'TotalAmount': 'sum',
            'TransactionID': 'count'
        }).reset_index()
        
        category_prefs.columns = ['CustomerID', 'Category', 'total_spent_category', 'purchases_category']
        
        category_spending = category_prefs.pivot(
            index='CustomerID', 
            columns='Category', 
            values='total_spent_category'
        ).fillna(0)
        
        category_purchases = category_prefs.pivot(
            index='CustomerID',
            columns='Category',
            values='purchases_category'
        ).fillna(0)
        
        category_spending_pct = category_spending.div(category_spending.sum(axis=1), axis=0)
        
        return category_spending_pct, category_purchases

    def create_time_aware_training_data(self, observation_days=90, prediction_days=14):
        """
        Create PROPER training data for promotion response prediction
        
        FIXES APPLIED:
        - FIX #1: Added product_category_encoded feature
        - FIX #4: Removed correlated features
        - No target leakage (features from past, target from future)
        """
        print("\n" + "=" * 70)
        print(" CREATING TIME-AWARE TRAINING DATA (ALL FIXES APPLIED)")
        print("=" * 70)
        
        min_date = self.transactions['TransactionDate'].min()
        max_date = self.transactions['TransactionDate'].max()
        
        print(f"\nData range: {min_date.date()} to {max_date.date()}")
        print(f"Observation window: {observation_days} days")
        print(f"Prediction window: {prediction_days} days")
        
        total_days = (max_date - min_date).days
        
        if total_days < observation_days + prediction_days:
            print("WARNING: Not enough data for proper time split!")
            observation_days = total_days // 3
            prediction_days = total_days // 6
        
        training_samples = []
        
        # Create observation points (every 30 days)
        observation_points = []
        current_point = min_date + timedelta(days=observation_days)
        while current_point + timedelta(days=prediction_days) <= max_date:
            observation_points.append(current_point)
            current_point += timedelta(days=30)
        
        print(f"\nCreating {len(observation_points)} observation points...")
        
        # Sample products for efficiency
        products = self.products['ProductID'].unique()
        if len(products) > 30:
            products = np.random.choice(products, 30, replace=False)
        
        for obs_idx, observation_date in enumerate(observation_points):
            print(f"  Processing observation point {obs_idx + 1}/{len(observation_points)}: {observation_date.date()}")
            
            obs_start = observation_date - timedelta(days=observation_days)
            obs_end = observation_date
            pred_start = observation_date
            pred_end = observation_date + timedelta(days=prediction_days)
            
            obs_transactions = self.transactions[
                (self.transactions['TransactionDate'] >= obs_start) &
                (self.transactions['TransactionDate'] < obs_end)
            ]
            
            pred_transactions = self.transactions[
                (self.transactions['TransactionDate'] >= pred_start) &
                (self.transactions['TransactionDate'] < pred_end)
            ]
            
            if len(obs_transactions) == 0:
                continue
            
            customer_features_obs = self.calculate_customer_features_for_window(
                obs_transactions, obs_end
            )
            
            category_affinity = self.calculate_category_affinity(obs_transactions)
            
            for customer_id in customer_features_obs['CustomerID'].unique():
                cust_feat = customer_features_obs[
                    customer_features_obs['CustomerID'] == customer_id
                ].iloc[0]
                
                cust_category = category_affinity.get(customer_id, {})
                
                for product_id in products:
                    prod_info = self.products[self.products['ProductID'] == product_id]
                    if len(prod_info) == 0:
                        continue
                    prod_info = prod_info.iloc[0]
                    
                    prod_category = prod_info['Category']
                    category_aff = cust_category.get(prod_category, 0)
                    
                    # FIX #1: Encode product category (use -1 for unknown)
                    if prod_category in self.category_encoder.classes_:
                        category_encoded = self.category_encoder.transform([prod_category])[0]
                    else:
                        category_encoded = -1  # Unknown category
                    
                    # Calculate product popularity in observation window only
                    prod_purchases_obs = len(obs_transactions[
                        obs_transactions['ProductID'] == product_id
                    ])
                    
                    # Category purchase count (NOT product-specific - no leakage)
                    category_purchase_count = len(obs_transactions[
                        (obs_transactions['CustomerID'] == customer_id) &
                        (obs_transactions['ProductID'].isin(
                            self.products[self.products['Category'] == prod_category]['ProductID']
                        ))
                    ])
                    
                    # TARGET: Future purchase (NO LEAKAGE)
                    purchased_in_future = len(pred_transactions[
                        (pred_transactions['CustomerID'] == customer_id) &
                        (pred_transactions['ProductID'] == product_id)
                    ]) > 0
                    
                    cust_age = self.customers[
                        self.customers['CustomerID'] == customer_id
                    ]['Age'].values
                    cust_age = cust_age[0] if len(cust_age) > 0 else 30
                    
                    sample = {
                        'CustomerID': customer_id,
                        'ProductID': product_id,
                        'observation_date': observation_date,
                        
                        # Customer features
                        'customer_purchase_frequency': cust_feat['purchase_frequency'],
                        'customer_avg_transaction': cust_feat['avg_transaction'],
                        'customer_recency': cust_feat['recency_days'],
                        'customer_promo_response_rate': cust_feat['promo_response_rate'],
                        'customer_age': cust_age,
                        
                        # Product features
                        'product_price': prod_info['Price'],
                        'product_category_encoded': category_encoded,  # FIX #1: Added
                        
                        # FIX #4: Removed product_popularity_obs (correlated with others)
                        
                        # Category features
                        'category_affinity': category_aff,
                        'category_purchase_count': category_purchase_count,
                        
                        # Target
                        'target': 1 if purchased_in_future else 0
                    }
                    
                    training_samples.append(sample)
        
        df = pd.DataFrame(training_samples)
        
        print(f"\n  Total samples: {len(df)}")
        if len(df) > 0:
            print(f"  Positive samples: {df['target'].sum()} ({df['target'].mean()*100:.2f}%)")
            print(f"  Negative samples: {len(df) - df['target'].sum()}")
        
        return df
    
    def calculate_customer_features_for_window(self, transactions, reference_date):
        """
        Calculate customer features for a specific time window.
        
        Public API for use by promotion_engine and other modules.
        
        Args:
            transactions: DataFrame of transactions to analyze
            reference_date: Date to calculate recency from
            
        Returns:
            DataFrame with customer features
        """
        
        customer_features = transactions.groupby('CustomerID').agg({
            'TransactionID': 'count',
            'TotalAmount': ['sum', 'mean'],
            'TransactionDate': lambda x: (reference_date - x.max()).days
        }).reset_index()
        
        customer_features.columns = ['CustomerID', 'purchase_frequency', 
                                     'total_spent', 'avg_transaction', 'recency_days']
        
        promo_trans = transactions[transactions['PromotionID'] != 'None']
        promo_response = promo_trans.groupby('CustomerID').size().reset_index(name='promo_purchases')
        customer_features = customer_features.merge(promo_response, on='CustomerID', how='left')
        customer_features['promo_purchases'] = customer_features['promo_purchases'].fillna(0)
        customer_features['promo_response_rate'] = (
            customer_features['promo_purchases'] / customer_features['purchase_frequency']
        )
        
        return customer_features
    
    def calculate_category_affinity(self, transactions):
        """
        Calculate customer's affinity for each category.
        
        Public API for use by promotion_engine and other modules.
        
        Args:
            transactions: DataFrame of transactions to analyze
            
        Returns:
            Dict mapping CustomerID -> {Category: affinity_score}
        """
        
        trans_with_cat = transactions.merge(
            self.products[['ProductID', 'Category']], on='ProductID'
        )
        
        cat_spending = trans_with_cat.groupby(['CustomerID', 'Category'])['TotalAmount'].sum().reset_index()
        
        total_spending = cat_spending.groupby('CustomerID')['TotalAmount'].transform('sum')
        cat_spending['affinity'] = cat_spending['TotalAmount'] / total_spending
        
        affinity_dict = {}
        for _, row in cat_spending.iterrows():
            if row['CustomerID'] not in affinity_dict:
                affinity_dict[row['CustomerID']] = {}
            affinity_dict[row['CustomerID']][row['Category']] = row['affinity']
        
        return affinity_dict
    
    def get_train_val_test_split(self, df, val_ratio=0.15, test_ratio=0.15):
        """
        FIX #2: Proper train/validation/test split
        - Train: Learn model parameters
        - Validation: Find optimal threshold (no leakage!)
        - Test: Final unbiased evaluation
        """
        print("\nCreating time-based train/validation/test split...")
        
        df = df.sort_values('observation_date')
        
        n = len(df)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"  Train samples: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
        print(f"  Validation samples: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
        print(f"  Test samples: {len(test_df)} ({len(test_df)/n*100:.1f}%)")
        
        if len(train_df) > 0 and len(val_df) > 0:
            train_end_date = train_df['observation_date'].max()
            val_end_date = val_df['observation_date'].max()
            print(f"  Train period: up to {train_end_date.date()}")
            print(f"  Validation period: {train_end_date.date()} to {val_end_date.date()}")
            print(f"  Test period: from {val_end_date.date()}")
        
        return train_df, val_df, test_df
    
    def get_time_based_train_test_split(self, df, test_ratio=0.25):
        """Legacy method for backward compatibility"""
        train_df, val_df, test_df = self.get_train_val_test_split(df, val_ratio=0, test_ratio=test_ratio)
        return train_df, test_df
    
    def create_training_data_for_promotion_targeting(self, product_id=None):
        """Legacy method - calls time-aware version"""
        return self.create_time_aware_training_data()
    
    def save_processed_data(self, output_dir='f:\\.1 Research\\Personalized Promotion Engine\\data\\processed'):
        """Save all processed datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nSaving processed datasets...")
        
        customer_features = self.create_customer_features()
        customer_features.to_csv(os.path.join(output_dir, 'customer_features.csv'), index=False)
        print(f"  * Saved customer_features.csv")
        
        product_features = self.create_product_features()
        product_features.to_csv(os.path.join(output_dir, 'product_features.csv'), index=False)
        print(f"  * Saved product_features.csv")
        
        interactions = self.create_customer_product_matrix()
        interactions.to_csv(os.path.join(output_dir, 'customer_product_interactions.csv'), index=False)
        print(f"  * Saved customer_product_interactions.csv")
        
        category_spending_pct, category_purchases = self.create_category_preferences()
        category_spending_pct.to_csv(os.path.join(output_dir, 'category_spending_pct.csv'))
        category_purchases.to_csv(os.path.join(output_dir, 'category_purchases.csv'))
        print(f"  * Saved category preference files")
        
        print(f"\n* All processed data saved to: {output_dir}")


def main():
    """Main function to run preprocessing"""
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    preprocessor.save_processed_data()
    
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
