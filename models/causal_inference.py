"""
Causal Inference for Promotion Impact
Determines if customers bought BECAUSE of promotion or would have bought anyway
This is advanced ML and adds significant research value
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class UpliftModel:
    """
    Uplift Modeling using Two-Model Approach
    Identifies customers who buy ONLY when promoted (persuadables)
    
    Customer Segments:
    1. Persuadables: Buy only if promoted → TARGET THESE!
    2. Sure Things: Buy anyway → Don't waste discount
    3. Lost Causes: Won't buy even with promotion → Skip
    4. Sleeping Dogs: Promotion makes them NOT buy → Avoid
    """
    
    def __init__(self):
        self.treatment_model = None  # Model for customers who received promotion
        self.control_model = None    # Model for customers who didn't
        
    def prepare_data(self, transactions_df, customers_df, products_df, target_product_id):
        """
        Prepare data for uplift modeling
        Need both promoted and non-promoted purchases
        """
        print("Preparing uplift modeling data...")
        
        # Filter for target product
        product_transactions = transactions_df[
            transactions_df['ProductID'] == target_product_id
        ].copy()
        
        # Create customer features
        customer_features = []
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['CustomerID']
            
            # Get customer's transactions
            cust_trans = transactions_df[transactions_df['CustomerID'] == customer_id]
            
            # Calculate features
            purchase_freq = len(cust_trans)
            avg_amount = cust_trans['TotalAmount'].mean() if len(cust_trans) > 0 else 0
            
            # Check if bought this product
            bought_product = len(product_transactions[
                product_transactions['CustomerID'] == customer_id
            ]) > 0
            
            # Check if bought with promotion
            bought_with_promo = len(product_transactions[
                (product_transactions['CustomerID'] == customer_id) &
                (product_transactions['PromotionID'] != 'None')
            ]) > 0
            
            # Treatment indicator
            received_promotion = len(transactions_df[
                (transactions_df['CustomerID'] == customer_id) &
                (transactions_df['ProductID'] == target_product_id) &
                (transactions_df['PromotionID'] != 'None')
            ]) > 0
            
            customer_features.append({
                'CustomerID': customer_id,
                'purchase_frequency': purchase_freq,
                'avg_transaction': avg_amount,
                'age': customer['Age'],
                'treatment': 1 if received_promotion else 0,  # Received promotion
                'outcome': 1 if bought_product else 0  # Bought product
            })
        
        df = pd.DataFrame(customer_features)
        
        print(f"  Total customers: {len(df)}")
        print(f"  Treatment group: {df['treatment'].sum()} ({df['treatment'].mean()*100:.1f}%)")
        print(f"  Purchase rate: {df['outcome'].sum()} ({df['outcome'].mean()*100:.1f}%)")
        
        return df
    
    def train(self, X_treatment, y_treatment, X_control, y_control):
        """
        Train two separate models: one for treatment, one for control
        """
        print("\nTraining uplift models...")
        
        # Model for treatment group (received promotion)
        self.treatment_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        self.treatment_model.fit(X_treatment, y_treatment)
        
        # Model for control group (no promotion)
        self.control_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        self.control_model.fit(X_control, y_control)
        
        print("✓ Uplift models trained")
        
    def predict_uplift(self, X):
        """
        Predict uplift (incremental effect of promotion)
        Uplift = P(buy|promotion) - P(buy|no promotion)
        """
        # Probability of buying with promotion
        p_treatment = self.treatment_model.predict_proba(X)[:, 1]
        
        # Probability of buying without promotion
        p_control = self.control_model.predict_proba(X)[:, 1]
        
        # Uplift is the difference
        uplift = p_treatment - p_control
        
        return uplift, p_treatment, p_control
    
    def segment_customers(self, X, customer_ids):
        """
        Segment customers into 4 groups based on uplift
        """
        uplift, p_treatment, p_control = self.predict_uplift(X)
        
        segments = []
        
        for i, customer_id in enumerate(customer_ids):
            u = uplift[i]
            pt = p_treatment[i]
            pc = p_control[i]
            
            # Determine segment
            if u > 0.1:  # High positive uplift
                if pc < 0.3:  # Low baseline probability
                    segment = 'Persuadable'  # TARGET THESE
                else:
                    segment = 'Sure Thing'  # Would buy anyway
            elif u < -0.1:  # Negative uplift
                segment = 'Sleeping Dog'  # Promotion hurts
            else:  # Low uplift
                if pt < 0.2:
                    segment = 'Lost Cause'  # Won't buy
                else:
                    segment = 'Sure Thing'
            
            segments.append({
                'CustomerID': customer_id,
                'uplift': u,
                'p_with_promotion': pt,
                'p_without_promotion': pc,
                'segment': segment,
                'should_target': segment == 'Persuadable'
            })
        
        return pd.DataFrame(segments)
    
    def calculate_incremental_roi(self, segments_df, product_price, discount_pct):
        """
        Calculate true incremental ROI from targeting persuadables only
        """
        persuadables = segments_df[segments_df['segment'] == 'Persuadable']
        
        if len(persuadables) == 0:
            return None
        
        # Incremental revenue (only from persuadables)
        incremental_conversions = persuadables['uplift'].sum()
        incremental_revenue = incremental_conversions * product_price
        
        # Cost (discounts given)
        discount_cost = len(persuadables) * product_price * (discount_pct / 100)
        
        # Incremental profit
        incremental_profit = incremental_revenue - discount_cost
        
        # ROI
        roi = (incremental_profit / discount_cost) * 100 if discount_cost > 0 else 0
        
        results = {
            'num_persuadables': len(persuadables),
            'incremental_conversions': incremental_conversions,
            'incremental_revenue': incremental_revenue,
            'discount_cost': discount_cost,
            'incremental_profit': incremental_profit,
            'incremental_roi': roi
        }
        
        return results


def compare_traditional_vs_causal(transactions_df, customers_df, products_df, product_id):
    """
    Compare traditional targeting vs causal targeting
    Shows the value of causal inference
    """
    print("\n" + "=" * 70)
    print(" TRADITIONAL vs CAUSAL TARGETING COMPARISON")
    print("=" * 70)
    
    # Initialize uplift model
    uplift_model = UpliftModel()
    
    # Prepare data
    data = uplift_model.prepare_data(transactions_df, customers_df, products_df, product_id)
    
    # Split into treatment and control
    treatment_data = data[data['treatment'] == 1]
    control_data = data[data['treatment'] == 0]
    
    feature_cols = ['purchase_frequency', 'avg_transaction', 'age']
    
    # Train uplift model
    if len(treatment_data) > 0 and len(control_data) > 0:
        uplift_model.train(
            treatment_data[feature_cols], treatment_data['outcome'],
            control_data[feature_cols], control_data['outcome']
        )
        
        # Segment all customers
        segments = uplift_model.segment_customers(
            data[feature_cols], data['CustomerID']
        )
        
        print("\n" + "-" * 70)
        print("CUSTOMER SEGMENTATION RESULTS")
        print("-" * 70)
        print(segments['segment'].value_counts())
        
        # Get product info
        product = products_df[products_df['ProductID'] == product_id].iloc[0]
        product_price = product['Price']
        
        # Calculate incremental ROI
        print("\n" + "-" * 70)
        print("INCREMENTAL ROI ANALYSIS")
        print("-" * 70)
        
        causal_results = uplift_model.calculate_incremental_roi(
            segments, product_price, discount_pct=15
        )
        
        if causal_results:
            print(f"\nCAUSAL TARGETING (Persuadables only):")
            print(f"  Customers Targeted: {causal_results['num_persuadables']}")
            print(f"  Incremental Conversions: {causal_results['incremental_conversions']:.1f}")
            print(f"  Incremental Revenue: Rs. {causal_results['incremental_revenue']:,.2f}")
            print(f"  Discount Cost: Rs. {causal_results['discount_cost']:,.2f}")
            print(f"  Incremental Profit: Rs. {causal_results['incremental_profit']:,.2f}")
            print(f"  Incremental ROI: {causal_results['incremental_roi']:.1f}%")
            
            # Compare with traditional (target top N by purchase probability)
            print(f"\n\nTRADITIONAL TARGETING (Top 100 by ML prediction):")
            top_100 = segments.nlargest(100, 'p_with_promotion')
            
            # This includes sure things who would buy anyway
            wasted_discounts = len(top_100[top_100['segment'] == 'Sure Thing'])
            
            print(f"  Customers Targeted: 100")
            print(f"  Includes Sure Things: {wasted_discounts} (wasted discounts)")
            print(f"  Includes Persuadables: {len(top_100[top_100['segment'] == 'Persuadable'])}")
            
            print("\n" + "=" * 70)
            print("CONCLUSION: Causal inference prevents wasting discounts on")
            print("customers who would buy anyway (Sure Things)")
            print("=" * 70)
        
        return segments, causal_results
    else:
        print("Not enough data for causal analysis")
        return None, None


def main():
    """Demo causal inference"""
    
    print("=" * 70)
    print(" CAUSAL INFERENCE FOR PROMOTIONS")
    print("=" * 70)
    
    # Load data
    from data_analysis.preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    # Select a product
    sample_product = preprocessor.products.sample(1).iloc[0]
    product_id = sample_product['ProductID']
    
    print(f"\nAnalyzing product: {sample_product['ProductName']}")
    print(f"Price: Rs. {sample_product['Price']:,.2f}")
    
    # Run comparison
    segments, results = compare_traditional_vs_causal(
        preprocessor.transactions,
        preprocessor.customers,
        preprocessor.products,
        product_id
    )
    
    if segments is not None:
        # Show sample persuadables
        persuadables = segments[segments['segment'] == 'Persuadable'].head(10)
        print("\n\nTop 10 Persuadable Customers (Best Targets):")
        print(persuadables[['CustomerID', 'uplift', 'p_with_promotion', 'p_without_promotion']])


if __name__ == "__main__":
    main()
