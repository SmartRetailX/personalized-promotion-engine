"""
Complete Personalized Promotion System
Integrates all models to provide end-to-end promotion targeting

FIX SYSTEM: CF + ML properly integrated
Pipeline: Product -> CF candidates -> ML scoring -> Top-N targets
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.purchase_prediction import PurchasePredictionModel
from models.collaborative_filtering import CollaborativeFilteringModel
from models.promotion_optimizer import PromotionOptimizer
from data_analysis.preprocessing import DataPreprocessor


class PersonalizedPromotionEngine:
    """
    Complete end-to-end promotion targeting system
    
    FIXED PIPELINE:
    Step 1: CF finds candidate customers (similar to product buyers)
    Step 2: ML predicts purchase probability for each candidate
    Step 3: Only customers above threshold receive promotion
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.purchase_model = None
        self.cf_model = None
        self.optimizer = None
        self.preprocessor = None
        
    def load_models(self):
        """Load all trained models"""
        print("Loading models...")
        
        # Load purchase prediction model
        purchase_model_path = os.path.join(self.models_dir, 'purchase_prediction_model.pkl')
        if os.path.exists(purchase_model_path):
            self.purchase_model = PurchasePredictionModel.load_model(purchase_model_path)
        else:
            print("  WARNING: Purchase prediction model not found. Train it first.")
        
        # Load collaborative filtering model
        cf_model_path = os.path.join(self.models_dir, 'collaborative_filtering_model.pkl')
        if os.path.exists(cf_model_path):
            self.cf_model = CollaborativeFilteringModel.load_model(cf_model_path)
        else:
            print("  WARNING: Collaborative filtering model not found. Train it first.")
        
        # Initialize optimizer
        self.optimizer = PromotionOptimizer()
        customer_features_path = 'data/processed/customer_features.csv'
        transactions_path = 'data/raw/Transactions.csv'
        
        if os.path.exists(customer_features_path) and os.path.exists(transactions_path):
            self.optimizer.load_data(customer_features_path, transactions_path)
        else:
            print("  WARNING: Optimizer data not found. Run preprocessing first.")
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_data()
        
        print("* Models loaded successfully\n")
    
    def generate_promotion_targets(self, product_id, top_n=50):
        """
        FIXED SYSTEM: Proper CF + ML pipeline
        
        Step 1: CF identifies candidate customers based on product/customer similarity
        Step 2: ML predicts purchase probability for each candidate
        Step 3: Only customers exceeding threshold receive promotion
        
        Returns: DataFrame with CustomerID, purchase_probability, targeting_method
        """
        print("\n" + "=" * 70)
        print(" GENERATING PROMOTION TARGETS (CF + ML PIPELINE)")
        print("=" * 70)
        
        # Get product info
        product = self.preprocessor.products[
            self.preprocessor.products['ProductID'] == product_id
        ]
        
        if len(product) == 0:
            print(f"Error: Product {product_id} not found")
            return pd.DataFrame()
        
        product = product.iloc[0]
        print(f"\nProduct: {product['ProductName']}")
        print(f"Category: {product['Category']}")
        print(f"Price: Rs. {product['Price']:,.2f}")
        
        # STEP 1: CF finds candidate customers WITH SCORES
        print("\n[STEP 1] Collaborative Filtering - Finding candidate customers...")
        cf_results = self.cf_model.find_customers_for_product(product_id, n=top_n * 3, return_scores=True)
        print(f"  Found {len(cf_results)} candidates from CF")
        
        # Build CF score lookup
        cf_scores = {}
        cf_candidates = []
        if len(cf_results) == 0:
            print("  WARNING: No candidates from CF, using all customers")
            cf_candidates = self.preprocessor.customers['CustomerID'].tolist()[:top_n * 3]
            cf_scores = {cid: 0.5 for cid in cf_candidates}  # Default score
        else:
            for item in cf_results:
                cf_candidates.append(item['CustomerID'])
                cf_scores[item['CustomerID']] = item['cf_score']
        
        # STEP 2: Prepare features for ML scoring
        print("\n[STEP 2] Preparing features for ML scoring...")
        
        # Get latest transaction date for feature calculation
        # Use longer window (180 days) to capture more customers from CF
        # CF was trained on historical data, so candidates may not be in last 90 days
        max_date = self.preprocessor.transactions['TransactionDate'].max()
        obs_start = max_date - pd.Timedelta(days=180)  # Extended from 90 to 180
        
        obs_transactions = self.preprocessor.transactions[
            self.preprocessor.transactions['TransactionDate'] >= obs_start
        ]
        
        # Calculate features for candidates
        customer_features_obs = self.preprocessor.calculate_customer_features_for_window(
            obs_transactions, max_date
        )
        
        category_affinity = self.preprocessor.calculate_category_affinity(obs_transactions)
        
        # Encode product category (use -1 for unknown)
        prod_category = product['Category']
        if prod_category in self.preprocessor.category_encoder.classes_:
            category_encoded = self.preprocessor.category_encoder.transform([prod_category])[0]
        else:
            category_encoded = -1  # Unknown category
            print(f"  Warning: Unknown category '{prod_category}', using -1")
        
        # Build feature dataframe for candidates
        candidate_features = []
        dropped_count = 0
        cold_customer_count = 0
        
        # Calculate fallback values using population medians (not zeros!)
        # This prevents cold customers from being scored as "worst possible"
        if len(customer_features_obs) > 0:
            fallback_frequency = customer_features_obs['purchase_frequency'].median()
            fallback_avg_txn = customer_features_obs['avg_transaction'].median()
            fallback_recency = customer_features_obs['recency_days'].median()
            fallback_promo_rate = customer_features_obs['promo_response_rate'].median()
        else:
            fallback_frequency = 5  # Reasonable defaults
            fallback_avg_txn = 500
            fallback_recency = 30
            fallback_promo_rate = 0.1
        
        for customer_id in cf_candidates:
            cust_feat = customer_features_obs[
                customer_features_obs['CustomerID'] == customer_id
            ]
            
            # FIX: Don't drop cold customers - use MEDIAN fallback features
            # Using population median instead of zeros prevents model from
            # rejecting all cold customers as "worst possible candidates"
            if len(cust_feat) == 0:
                # Cold customer: no recent transactions but CF thinks they're relevant
                cold_customer_count += 1
                cust_feat_dict = {
                    'purchase_frequency': fallback_frequency,
                    'avg_transaction': fallback_avg_txn,
                    'recency_days': fallback_recency,  # Use median, not 999
                    'promo_response_rate': fallback_promo_rate
                }
            else:
                cust_feat_dict = {
                    'purchase_frequency': cust_feat.iloc[0]['purchase_frequency'],
                    'avg_transaction': cust_feat.iloc[0]['avg_transaction'],
                    'recency_days': cust_feat.iloc[0]['recency_days'],
                    'promo_response_rate': cust_feat.iloc[0]['promo_response_rate']
                }
            
            cust_category = category_affinity.get(customer_id, {})
            
            # Get customer age
            cust_age = self.preprocessor.customers[
                self.preprocessor.customers['CustomerID'] == customer_id
            ]['Age'].values
            cust_age = cust_age[0] if len(cust_age) > 0 else 30
            
            # Category purchase count
            category_purchase_count = len(obs_transactions[
                (obs_transactions['CustomerID'] == customer_id) &
                (obs_transactions['ProductID'].isin(
                    self.preprocessor.products[
                        self.preprocessor.products['Category'] == prod_category
                    ]['ProductID']
                ))
            ])
            
            features = {
                'CustomerID': customer_id,
                'ProductID': product_id,
                'customer_purchase_frequency': cust_feat_dict['purchase_frequency'],
                'customer_avg_transaction': cust_feat_dict['avg_transaction'],
                'customer_recency': cust_feat_dict['recency_days'],
                'customer_promo_response_rate': cust_feat_dict['promo_response_rate'],
                'customer_age': cust_age,
                'product_price': product['Price'],
                'product_category_encoded': category_encoded,
                'category_affinity': cust_category.get(prod_category, 0),
                'category_purchase_count': category_purchase_count,
                'cf_score': cf_scores.get(customer_id, 0),  # CF similarity score
            }
            
            candidate_features.append(features)
        
        if len(candidate_features) == 0:
            print("  ERROR: No valid candidates with features")
            return pd.DataFrame()
        
        if cold_customer_count > 0:
            print(f"  Note: {cold_customer_count} cold customers (no recent activity) included with fallback features")
        
        candidate_df = pd.DataFrame(candidate_features)
        print(f"  Prepared features for {len(candidate_df)} candidates")
        
        # STEP 3: ML scoring
        print("\n[STEP 3] ML Model - Scoring candidates...")
        
        X, _, _ = self.purchase_model.prepare_features(candidate_df)
        
        # Validate feature alignment with training
        if self.purchase_model.feature_cols is not None:
            expected_features = set(self.purchase_model.feature_cols)
            actual_features = set(X.columns)
            
            if expected_features != actual_features:
                missing = expected_features - actual_features
                extra = actual_features - expected_features
                error_msg = "Feature mismatch between training and inference!\n"
                if missing:
                    error_msg += f"  Missing features: {missing}\n"
                if extra:
                    error_msg += f"  Extra features: {extra}\n"
                raise ValueError(error_msg)
        
        candidate_df['purchase_probability'] = self.purchase_model.predict_proba(X)
        
        # Create hybrid score: combine ML probability with CF similarity
        # ML is the primary signal (purchase likelihood), CF adds diversity
        ML_WEIGHT = 0.7
        CF_WEIGHT = 0.3
        candidate_df['hybrid_score'] = (
            ML_WEIGHT * candidate_df['purchase_probability'] + 
            CF_WEIGHT * candidate_df['cf_score']
        )
        
        print(f"  Hybrid scoring: {ML_WEIGHT:.0%} ML + {CF_WEIGHT:.0%} CF")
        
        # Debug: Show probability distribution
        print(f"  ML probability range: {candidate_df['purchase_probability'].min():.4f} - {candidate_df['purchase_probability'].max():.4f}")
        print(f"  ML probability median: {candidate_df['purchase_probability'].median():.4f}")
        
        # Apply threshold using PERCENTILE approach instead of fixed threshold
        # The F1-optimal threshold assumes we want precision=recall balance
        # For promotion targeting, we want to capture the TOP candidates regardless of absolute probability
        model_threshold = self.purchase_model.optimal_threshold
        
        # Use percentile-based threshold: target the top 50% of candidates
        # This ensures we always get some targets, even if absolute probabilities are low
        percentile_threshold = candidate_df['purchase_probability'].quantile(0.5)
        
        # Also set a minimum based on random baseline (1.5% positive rate means random=0.015)
        # Any probability above random baseline is worth considering
        min_useful_threshold = 0.02  # Just above random
        
        # Use the higher of percentile or minimum useful threshold
        promotion_threshold = max(min_useful_threshold, percentile_threshold)
        
        print(f"  Model threshold: {model_threshold:.4f} (F1-optimal, not used for filtering)")
        print(f"  Percentile threshold (50th): {percentile_threshold:.4f}")
        print(f"  Promotion threshold: {promotion_threshold:.4f} (marketing-optimized)")
        
        # STEP 4: Filter and rank
        print("\n[STEP 4] Filtering and ranking by hybrid score...")
        
        # Filter to only eligible customers (ML probability above promotion threshold)
        eligible_df = candidate_df[candidate_df['purchase_probability'] >= promotion_threshold]
        below_threshold_count = len(candidate_df) - len(eligible_df)
        
        # Sort by HYBRID score (not just ML probability)
        eligible_df = eligible_df.sort_values('hybrid_score', ascending=False)
        
        # Take top N from eligible only
        top_targets = eligible_df.head(top_n).copy()
        
        # Add targeting info
        top_targets['above_threshold'] = True  # All returned customers are above threshold
        top_targets['targeting_method'] = 'cf_ml_hybrid'
        
        print(f"\n  Candidates scored: {len(candidate_df)}")
        print(f"  Above threshold (eligible): {len(eligible_df)}")
        print(f"  Below threshold (filtered out): {below_threshold_count}")
        print(f"  Returned for targeting: {len(top_targets)}")
        
        if len(top_targets) < top_n and len(eligible_df) < top_n:
            print(f"  Note: Only {len(eligible_df)} customers meet threshold, requested {top_n}")
        
        print("\n  Top 5 targets (all eligible for promotion):")
        for idx, row in top_targets.head(5).iterrows():
            print(f"    {row['CustomerID']}: ML={row['purchase_probability']:.2%}, CF={row['cf_score']:.2f}, Hybrid={row['hybrid_score']:.2%}")
        
        return top_targets[['CustomerID', 'ProductID', 'purchase_probability', 'cf_score',
                           'hybrid_score', 'above_threshold', 'category_affinity', 'targeting_method']]
        
    def get_promotion_targets(self, product_id, strategy='hybrid', top_n=100):
        """
        Get target customers for a product promotion
        
        Strategies:
        - 'ml_only': Use only purchase prediction model
        - 'cf_only': Use only collaborative filtering
        - 'hybrid': Combine both approaches (recommended)
        """
        print(f"\nGetting promotion targets for product: {product_id}")
        print(f"Strategy: {strategy}, Top N: {top_n}")
        
        if strategy == 'ml_only':
            return self._get_targets_ml(product_id, top_n)
        elif strategy == 'cf_only':
            return self._get_targets_cf(product_id, top_n)
        else:
            return self._get_targets_hybrid(product_id, top_n)
    
    def _get_targets_ml(self, product_id, top_n):
        """Get targets using ML prediction model"""
        # Get product category for targeting
        product = self.preprocessor.products[
            self.preprocessor.products['ProductID'] == product_id
        ]
        if len(product) == 0:
            print(f"  Warning: Product {product_id} not found")
            return pd.DataFrame(columns=['CustomerID', 'purchase_probability'])
        
        product_category = product.iloc[0]['Category']
        
        # Create training data - this creates samples for all customer-product pairs
        training_data = self.preprocessor.create_training_data_for_promotion_targeting(product_id)
        
        if len(training_data) == 0:
            print(f"  Warning: No training data generated")
            return pd.DataFrame(columns=['CustomerID', 'purchase_probability'])
        
        # Filter to products in the SAME CATEGORY (not just specific product)
        # This gives us more data while still being relevant
        category_products = self.preprocessor.products[
            self.preprocessor.products['Category'] == product_category
        ]['ProductID'].tolist()
        
        category_data = training_data[
            training_data['ProductID'].isin(category_products)
        ].copy()
        
        if len(category_data) == 0:
            print(f"  Warning: No data for category {product_category}")
            return pd.DataFrame(columns=['CustomerID', 'purchase_probability'])
        
        print(f"  Scoring customers for category: {product_category} ({len(category_data)} samples)")
        
        # Score all customers using ML model
        X, _, data = self.purchase_model.prepare_features(category_data)
        data['purchase_probability'] = self.purchase_model.predict_proba(X)
        
        # Aggregate by customer (take max probability across category products)
        customer_scores = data.groupby('CustomerID').agg({
            'purchase_probability': 'max',
            'customer_promo_response_rate': 'first',
            'category_affinity': 'first'
        }).reset_index()
        
        # Add product ID for consistency
        customer_scores['ProductID'] = product_id
        
        # Sort and return top N
        top_customers = customer_scores.nlargest(top_n, 'purchase_probability')
        
        return top_customers
    
    def _get_targets_cf(self, product_id, top_n):
        """Get targets using collaborative filtering with scores"""
        # Find customers who bought similar products (with scores)
        cf_results = self.cf_model.find_customers_for_product(
            product_id, top_n * 2, return_scores=True
        )
        
        if not cf_results:
            return pd.DataFrame(columns=['CustomerID', 'cf_score', 'targeting_method'])
        
        # Create DataFrame with CF scores
        df = pd.DataFrame(cf_results[:top_n])
        df['targeting_method'] = 'collaborative_filtering'
        
        return df
    
    def _get_targets_hybrid(self, product_id, top_n):
        """Hybrid approach: combine ML and CF with proper scoring"""
        # Get ML predictions
        ml_targets = self._get_targets_ml(product_id, top_n * 2)
        
        # Get CF targets with scores
        cf_targets = self._get_targets_cf(product_id, top_n * 2)
        
        # Build CF score lookup
        cf_score_lookup = {}
        if len(cf_targets) > 0 and 'cf_score' in cf_targets.columns:
            cf_score_lookup = dict(zip(cf_targets['CustomerID'], cf_targets['cf_score']))
        
        # Handle empty ML results - fall back to CF only
        if len(ml_targets) == 0 or 'CustomerID' not in ml_targets.columns:
            print("  Note: ML returned no results, using CF-only targeting")
            if len(cf_targets) == 0:
                print("  WARNING: Both ML and CF returned no results!")
                return pd.DataFrame(columns=['CustomerID', 'purchase_probability', 'cf_score', 'targeting_method'])
            # Return CF results with estimated probability
            cf_targets['purchase_probability'] = cf_targets['cf_score'] * 0.5  # Conservative estimate
            cf_targets['targeting_method'] = 'cf_only_fallback'
            return cf_targets.head(top_n)
        
        # Combine: customers from both methods get higher priority
        ml_customers = set(ml_targets['CustomerID'].tolist())
        cf_customers = set(cf_targets['CustomerID'].tolist())
        
        # Customers in both
        both = ml_customers & cf_customers
        
        # Customers only in ML
        only_ml = ml_customers - cf_customers
        
        # Customers only in CF
        only_cf = cf_customers - ml_customers
        
        # Combine with priority
        combined = []
        
        # Add customers from both (highest priority)
        for cust_id in both:
            ml_row = ml_targets[ml_targets['CustomerID'] == cust_id]
            if len(ml_row) > 0:
                combined.append({
                    'CustomerID': cust_id,
                    'purchase_probability': ml_row.iloc[0]['purchase_probability'],
                    'cf_score': cf_score_lookup.get(cust_id, 0.5),
                    'targeting_method': 'hybrid_both'
                })
        
        # Add ML-only customers
        for cust_id in only_ml:
            ml_row = ml_targets[ml_targets['CustomerID'] == cust_id]
            if len(ml_row) > 0:
                combined.append({
                    'CustomerID': cust_id,
                    'purchase_probability': ml_row.iloc[0]['purchase_probability'],
                    'cf_score': 0,  # Not found by CF
                    'targeting_method': 'ml_only'
                })
        
        # Add CF-only customers (use CF score to estimate probability)
        for cust_id in only_cf:
            cf_score = cf_score_lookup.get(cust_id, 0.5)
            # Estimate probability based on CF similarity (scaled down)
            estimated_prob = 0.5 * cf_score  # More conservative estimate
            combined.append({
                'CustomerID': cust_id,
                'purchase_probability': estimated_prob,
                'cf_score': cf_score,
                'targeting_method': 'cf_only'
            })
        
        df = pd.DataFrame(combined)
        # Sort by purchase_probability for consistent ranking
        df = df.sort_values('purchase_probability', ascending=False).head(top_n)
        
        return df
    
    def create_promotion_campaign(self, product_id, max_targets=100, 
                                 strategy='hybrid', optimize=True):
        """
        Create a complete promotion campaign
        
        Returns:
        - Target customer list with personalized discounts
        - Campaign summary with expected metrics
        """
        print("\n" + "=" * 70)
        print(" CREATING PROMOTION CAMPAIGN")
        print("=" * 70)
        
        # Get product info
        product = self.preprocessor.products[
            self.preprocessor.products['ProductID'] == product_id
        ]
        
        if len(product) == 0:
            print(f"Error: Product {product_id} not found")
            return None, None
        
        product = product.iloc[0]
        product_name = product['ProductName']
        product_price = product['Price']
        
        print(f"\nProduct: {product_name}")
        print(f"Price: Rs. {product_price:,.2f}")
        
        # Step 1: Get target customers
        print(f"\n1. Finding target customers (strategy: {strategy})...")
        target_customers = self.get_promotion_targets(product_id, strategy, max_targets * 2)
        
        print(f"   Found {len(target_customers)} potential customers")
        
        # Step 2: Optimize if requested
        if optimize and self.optimizer:
            print("\n2. Optimizing campaign...")
            campaign, summary = self.optimizer.generate_promotion_campaign(
                product_id, product_name, product_price,
                target_customers, max_targets
            )
        else:
            # No optimization - just return customers ranked by purchase probability
            # Discount will be set by the caller (user-defined)
            print("\n2. Returning top customers by purchase probability...")
            campaign = target_customers.head(max_targets).copy()
            # Note: Discount is NOT set here - it's defined by the user in campaign generator
            
            summary = {
                'product_id': product_id,
                'product_name': product_name,
                'num_customers_targeted': len(campaign)
            }
        
        print("\n" + "=" * 70)
        print(" CAMPAIGN CREATED SUCCESSFULLY")
        print("=" * 70)
        
        return campaign, summary
    
    def _calculate_historical_conversion_rate(self, product_id=None):
        """
        Calculate historical conversion rate from actual data.
        
        If product_id is provided, calculates for that product's category.
        Otherwise calculates overall promotion conversion rate.
        """
        transactions = self.preprocessor.transactions
        
        # Promotion transactions
        promo_trans = transactions[transactions['PromotionID'] != 'None']
        
        if len(promo_trans) == 0:
            return 0.10  # Fallback to 10% if no promo data
        
        if product_id:
            # Get product category
            product = self.preprocessor.products[
                self.preprocessor.products['ProductID'] == product_id
            ]
            if len(product) > 0:
                category = product.iloc[0]['Category']
                category_products = self.preprocessor.products[
                    self.preprocessor.products['Category'] == category
                ]['ProductID'].tolist()
                
                # Category-specific conversion
                category_promo_trans = promo_trans[promo_trans['ProductID'].isin(category_products)]
                if len(category_promo_trans) > 0:
                    # Unique customers who purchased in category with promotion
                    promo_buyers = category_promo_trans['CustomerID'].nunique()
                    total_customers = transactions['CustomerID'].nunique()
                    return promo_buyers / total_customers
        
        # Overall conversion rate: customers who bought with promotion / total customers
        promo_buyers = promo_trans['CustomerID'].nunique()
        total_customers = transactions['CustomerID'].nunique()
        
        return promo_buyers / total_customers
    
    def compare_personalized_vs_broadcast(self, product_id, broadcast_discount=15, 
                                          conversion_rate=None):
        """
        Compare personalized promotion vs broadcast to all customers
        Key research metric
        
        Args:
            product_id: Product to compare campaigns for
            broadcast_discount: Discount percentage for broadcast campaign
            conversion_rate: Override conversion rate (if None, calculated from data)
        """
        print("\n" + "=" * 70)
        print(" PERSONALIZED vs BROADCAST COMPARISON")
        print("=" * 70)
        
        product = self.preprocessor.products[
            self.preprocessor.products['ProductID'] == product_id
        ].iloc[0]
        
        product_price = product['Price']
        
        # Calculate or use provided conversion rate
        if conversion_rate is None:
            conversion_rate = self._calculate_historical_conversion_rate(product_id)
        print(f"\nUsing conversion rate: {conversion_rate:.1%} (from historical data)")
        
        # Personalized campaign
        print("\nA. PERSONALIZED CAMPAIGN (Top 100 customers)")
        personalized_campaign, personalized_summary = self.create_promotion_campaign(
            product_id, max_targets=100, optimize=True
        )
        
        # Broadcast to all customers
        print("\n\nB. BROADCAST CAMPAIGN (All 1000 customers)")
        all_customers = self.preprocessor.customers['CustomerID'].tolist()
        
        # Data-driven conversion rate instead of hardcoded 0.1
        broadcast_discount_cost = len(all_customers) * product_price * (broadcast_discount / 100) * conversion_rate
        broadcast_revenue = len(all_customers) * product_price * conversion_rate
        broadcast_profit = broadcast_revenue - broadcast_discount_cost
        
        print(f"   Customers Reached: {len(all_customers)}")
        print(f"   Discount: {broadcast_discount}%")
        print(f"   Conversion Rate: {conversion_rate:.1%}")
        print(f"   Expected Cost: Rs. {broadcast_discount_cost:,.2f}")
        print(f"   Expected Revenue: Rs. {broadcast_revenue:,.2f}")
        print(f"   Expected Profit: Rs. {broadcast_profit:,.2f}")
        
        # Comparison
        print("\n" + "-" * 70)
        print("COMPARISON SUMMARY")
        print("-" * 70)
        
        if personalized_summary and 'total_expected_profit' in personalized_summary:
            profit_improvement = (
                (personalized_summary['total_expected_profit'] - broadcast_profit) / 
                (broadcast_profit + 1) * 100
            )
            
            efficiency = (
                personalized_summary['num_customers_targeted'] / len(all_customers) * 100
            )
            
            print(f"Personalized reaches {efficiency:.1f}% of customers")
            print(f"Profit improvement: {profit_improvement:.1f}%")
            print(f"Cost efficiency: {100 - efficiency:.1f}% reduction in marketing costs")
        
        print("\n" + "=" * 70)
        
        return {
            'personalized': personalized_summary,
            'broadcast': {
                'cost': broadcast_discount_cost,
                'revenue': broadcast_revenue,
                'profit': broadcast_profit
            }
        }


def main():
    """Demo of the complete system with FIXED CF+ML pipeline"""
    
    print("=" * 70)
    print(" PERSONALIZED PROMOTION ENGINE - DEMO")
    print(" FIX: CF + ML properly integrated pipeline")
    print("=" * 70)
    
    # Initialize engine
    engine = PersonalizedPromotionEngine()
    engine.load_models()
    
    # Get a sample product
    sample_product = engine.preprocessor.products[
        engine.preprocessor.products['Category'] == 'Bakery'
    ].iloc[0]
    
    product_id = sample_product['ProductID']
    
    # =====================================================
    # METHOD 1: NEW PIPELINE (Recommended)
    # CF finds candidates -> ML scores -> Threshold filters
    # =====================================================
    print("\n" + "=" * 70)
    print(" METHOD 1: CF + ML PIPELINE (RECOMMENDED)")
    print("=" * 70)
    
    targets = engine.generate_promotion_targets(product_id, top_n=50)
    
    if len(targets) > 0:
        print("\n" + "-" * 50)
        print("TARGET SUMMARY:")
        print("-" * 50)
        
        above = targets[targets['above_threshold'] == True]
        below = targets[targets['above_threshold'] == False]
        
        print(f"  Total targets: {len(targets)}")
        print(f"  SEND promotion (above threshold): {len(above)}")
        print(f"  SKIP promotion (below threshold): {len(below)}")
        
        if len(above) > 0:
            print(f"\n  Average probability (SEND): {above['purchase_probability'].mean():.2%}")
            print(f"  Min probability (SEND): {above['purchase_probability'].min():.2%}")
        
        print("\n  Customers to receive promotion:")
        for _, row in above.head(10).iterrows():
            print(f"    {row['CustomerID']}: {row['purchase_probability']:.2%}")
    
    # =====================================================
    # METHOD 2: Legacy method with optimization
    # =====================================================
    print("\n\n" + "=" * 70)
    print(" METHOD 2: LEGACY WITH OPTIMIZATION")
    print("=" * 70)
    
    campaign, summary = engine.create_promotion_campaign(
        product_id, max_targets=50, strategy='hybrid', optimize=True
    )
    
    # Show top 10 targets
    if campaign is not None and len(campaign) > 0:
        print("\nTop 10 Target Customers:")
        cols_to_show = ['CustomerID', 'purchase_probability']
        if 'optimal_discount' in campaign.columns:
            cols_to_show.append('optimal_discount')
        if 'expected_roi' in campaign.columns:
            cols_to_show.append('expected_roi')
        print(campaign.head(10)[cols_to_show])
    
    print("\n" + "=" * 70)
    print(" DEMO COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
