"""
Promotion Optimizer - Advanced Feature
Uses optimization techniques to:
1. Determine optimal discount percentage for each customer
2. Predict promotion ROI before sending
3. Prevent promotion fatigue
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os


class PromotionOptimizer:
    """
    Advanced promotion optimization engine
    This adds research value beyond basic targeting
    """
    
    def __init__(self):
        self.customer_features = None
        self.promotion_history = None
        self.fatigue_thresholds = {
            'frequent_shoppers': 4,  # Max promotions per month
            'regular_shoppers': 3,
            'occasional_shoppers': 2,
            'rare_shoppers': 1
        }
        
    def load_data(self, customer_features_path, transactions_path):
        """Load necessary data"""
        self.customer_features = pd.read_csv(customer_features_path)
        transactions = pd.read_csv(transactions_path)
        
        # Extract promotion history
        self.promotion_history = transactions[transactions['PromotionID'] != 'None'].copy()
        self.promotion_history['TransactionDate'] = pd.to_datetime(self.promotion_history['TransactionDate'])
        
    def calculate_optimal_discount(self, customer_id, product_price, base_discount=15):
        """
        Calculate optimal discount percentage for a customer
        Based on their price sensitivity and historical response
        """
        customer = self.customer_features[self.customer_features['CustomerID'] == customer_id]
        
        if len(customer) == 0:
            return base_discount
        
        customer = customer.iloc[0]
        
        # Factor 1: Promotion response rate
        promo_response = customer['promo_response_rate']
        
        # Factor 2: Price sensitivity (based on average discount received)
        if customer['avg_discount_per_transaction'] > 0:
            typical_discount = (customer['avg_discount_per_transaction'] / 
                               customer['avg_transaction_value']) * 100
        else:
            typical_discount = base_discount
        
        # Factor 3: Customer value (high-value customers need less discount)
        if customer['total_spent'] > self.customer_features['total_spent'].quantile(0.75):
            value_multiplier = 0.8  # Give less discount to high-value customers
        elif customer['total_spent'] < self.customer_features['total_spent'].quantile(0.25):
            value_multiplier = 1.2  # Give more discount to low-value customers
        else:
            value_multiplier = 1.0
        
        # Calculate optimal discount
        if promo_response > 0.5:
            # Highly responsive - can give lower discount
            optimal = base_discount * 0.8 * value_multiplier
        elif promo_response > 0.2:
            # Moderately responsive
            optimal = base_discount * value_multiplier
        else:
            # Low response - need higher discount to convert
            optimal = base_discount * 1.3 * value_multiplier
        
        # Round to nearest 5
        optimal = round(optimal / 5) * 5
        
        # Constraints
        optimal = max(5, min(50, optimal))  # Between 5% and 50%
        
        return optimal
    
    def check_promotion_fatigue(self, customer_id, current_date=None):
        """
        Check if customer has received too many promotions recently
        Prevents promotion fatigue
        """
        if current_date is None:
            current_date = datetime.now()
        
        # Get customer segment
        customer = self.customer_features[self.customer_features['CustomerID'] == customer_id]
        if len(customer) == 0:
            return False  # Unknown customer, allow promotion
        
        segment = customer.iloc[0]['CustomerSegment']
        threshold = self.fatigue_thresholds.get(segment, 2)
        
        # Count promotions in last 30 days
        one_month_ago = current_date - timedelta(days=30)
        recent_promos = self.promotion_history[
            (self.promotion_history['CustomerID'] == customer_id) &
            (self.promotion_history['TransactionDate'] >= one_month_ago)
        ]
        
        promo_count = len(recent_promos)
        
        is_fatigued = promo_count >= threshold
        
        return is_fatigued, promo_count, threshold
    
    def predict_promotion_roi(self, customer_id, product_id, product_price, 
                             discount_percentage, predicted_purchase_probability):
        """
        Predict ROI of sending a promotion
        ROI = (Expected Revenue - Cost) / Cost
        """
        # Cost = discount amount if customer purchases
        discount_amount = product_price * (discount_percentage / 100)
        
        # Expected revenue = purchase probability * full price
        expected_revenue = predicted_purchase_probability * product_price
        
        # Expected cost = purchase probability * discount
        expected_cost = predicted_purchase_probability * discount_amount
        
        # Expected profit
        expected_profit = expected_revenue - expected_cost
        
        # ROI
        if expected_cost > 0:
            roi = (expected_profit / expected_cost) * 100
        else:
            roi = 0
        
        return {
            'expected_revenue': expected_revenue,
            'expected_cost': expected_cost,
            'expected_profit': expected_profit,
            'roi_percentage': roi,
            'should_send': roi > 50  # Only send if ROI > 50%
        }
    
    def rank_customers_for_promotion(self, customer_list, product_id, product_price,
                                    purchase_probabilities, current_date=None):
        """
        Rank customers considering:
        1. Purchase probability
        2. Promotion fatigue
        3. Expected ROI
        4. Customer lifetime value
        """
        ranked_customers = []
        
        for i, customer_id in enumerate(customer_list):
            # Check fatigue
            is_fatigued, promo_count, threshold = self.check_promotion_fatigue(customer_id, current_date)
            
            if is_fatigued:
                continue  # Skip fatigued customers
            
            # Calculate optimal discount
            optimal_discount = self.calculate_optimal_discount(customer_id, product_price)
            
            # Predict ROI
            purchase_prob = purchase_probabilities[i] if i < len(purchase_probabilities) else 0.5
            roi_info = self.predict_promotion_roi(
                customer_id, product_id, product_price, optimal_discount, purchase_prob
            )
            
            if not roi_info['should_send']:
                continue  # Skip if ROI is too low
            
            # Get customer value
            customer = self.customer_features[self.customer_features['CustomerID'] == customer_id]
            if len(customer) > 0:
                customer_value = customer.iloc[0]['total_spent']
            else:
                customer_value = 0
            
            # Calculate composite score
            # Weight: 50% purchase probability, 30% ROI, 20% customer value
            normalized_value = customer_value / (self.customer_features['total_spent'].max() + 1)
            composite_score = (
                0.5 * purchase_prob +
                0.3 * (roi_info['roi_percentage'] / 200) +  # Normalize ROI
                0.2 * normalized_value
            )
            
            ranked_customers.append({
                'CustomerID': customer_id,
                'purchase_probability': purchase_prob,
                'optimal_discount': optimal_discount,
                'expected_roi': roi_info['roi_percentage'],
                'customer_value': customer_value,
                'composite_score': composite_score,
                'recent_promo_count': promo_count
            })
        
        # Sort by composite score
        ranked_customers.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return pd.DataFrame(ranked_customers)
    
    def generate_promotion_campaign(self, product_id, product_name, product_price,
                                   target_customers_df, max_customers=100):
        """
        Generate a complete promotion campaign
        Returns optimized list of customers with personalized discounts
        """
        print(f"\nGenerating promotion campaign for {product_name}...")
        
        # Extract data
        customer_ids = target_customers_df['CustomerID'].tolist()
        purchase_probs = target_customers_df['purchase_probability'].tolist()
        
        # Rank customers
        ranked = self.rank_customers_for_promotion(
            customer_ids, product_id, product_price, purchase_probs
        )
        
        # Limit to max customers
        campaign = ranked.head(max_customers)
        
        # Add campaign summary
        total_expected_revenue = (campaign['purchase_probability'] * product_price).sum()
        total_expected_cost = (
            campaign['purchase_probability'] * 
            product_price * 
            (campaign['optimal_discount'] / 100)
        ).sum()
        total_expected_profit = total_expected_revenue - total_expected_cost
        
        campaign_summary = {
            'product_id': product_id,
            'product_name': product_name,
            'product_price': product_price,
            'num_customers_targeted': len(campaign),
            'avg_discount': campaign['optimal_discount'].mean(),
            'total_expected_revenue': total_expected_revenue,
            'total_expected_cost': total_expected_cost,
            'total_expected_profit': total_expected_profit,
            'avg_roi': campaign['expected_roi'].mean()
        }
        
        print(f"  Targeted Customers: {campaign_summary['num_customers_targeted']}")
        print(f"  Average Discount: {campaign_summary['avg_discount']:.1f}%")
        print(f"  Expected Revenue: Rs. {campaign_summary['total_expected_revenue']:,.2f}")
        print(f"  Expected Cost: Rs. {campaign_summary['total_expected_cost']:,.2f}")
        print(f"  Expected Profit: Rs. {campaign_summary['total_expected_profit']:,.2f}")
        print(f"  Average ROI: {campaign_summary['avg_roi']:.1f}%")
        
        return campaign, campaign_summary


def main():
    """Example usage"""
    
    print("=" * 70)
    print(" PROMOTION OPTIMIZER - DEMO")
    print("=" * 70)
    
    # Initialize optimizer
    optimizer = PromotionOptimizer()
    
    # Load data
    print("\nLoading data...")
    customer_features_path = 'f:\\.1 Research\\Personalized Promotion Engine\\data\\processed\\customer_features.csv'
    transactions_path = 'f:\\.1 Research\\Personalized Promotion Engine\\data\\raw\\Transactions.csv'
    
    optimizer.load_data(customer_features_path, transactions_path)
    
    # Example: Check optimal discount for a customer
    sample_customer = optimizer.customer_features.iloc[0]['CustomerID']
    optimal_discount = optimizer.calculate_optimal_discount(sample_customer, product_price=500)
    
    print(f"\nSample Analysis for Customer: {sample_customer}")
    print(f"  Optimal Discount: {optimal_discount}%")
    
    # Check fatigue
    is_fatigued, count, threshold = optimizer.check_promotion_fatigue(sample_customer)
    print(f"  Promotion Fatigue: {'Yes' if is_fatigued else 'No'}")
    print(f"  Recent Promotions: {count}/{threshold}")
    
    print("\n" + "=" * 70)
    print(" PROMOTION OPTIMIZER READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
