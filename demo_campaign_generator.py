"""
Practical Demonstration: Personalized Promotion Campaign Generator
This script shows how to USE the trained models to generate real customer lists
for specific product promotions.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.promotion_engine import PersonalizedPromotionEngine
from models.collaborative_filtering import CollaborativeFilteringModel

class CampaignGenerator:
    """Generate actual promotion campaigns with customer lists"""
    
    def __init__(self):
        self.engine = PersonalizedPromotionEngine()
        self.cf_model = CollaborativeFilteringModel()
        self.output_dir = "campaign_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_models(self):
        """Load trained models"""
        print("Loading trained models...")
        self.engine.load_models()
        self.cf_model.load_model()
        print("* Models loaded successfully\n")
        
    def show_available_products(self, category=None):
        """Show products available for promotion"""
        products = self.engine.preprocessor.products.copy()
        
        if category:
            products = products[products['Category'] == category]
            
        print(f"\n{'='*70}")
        print(f" AVAILABLE PRODUCTS" + (f" - {category}" if category else ""))
        print(f"{'='*70}")
        
        for idx, row in products.head(20).iterrows():
            print(f"{row['ProductID']}: {row['ProductName']:<40} Rs. {row['Price']:.2f} ({row['Category']})")
        
        print(f"\nTotal products: {len(products)}")
        return products
    
    def generate_promotion_campaign(self, product_id, discount_percent=10, max_customers=100):
        """
        Generate a promotion campaign for a specific product
        Returns: CSV file with customer list to send promotions
        """
        print(f"\n{'='*70}")
        print(f" GENERATING PROMOTION CAMPAIGN")
        print(f"{'='*70}\n")
        
        # Get product details
        product = self.engine.preprocessor.products[
            self.engine.preprocessor.products['ProductID'] == product_id
        ].iloc[0]
        
        print(f"Product: {product['ProductName']}")
        print(f"Category: {product['Category']}")
        print(f"Price: Rs. {product['Price']:.2f}")
        print(f"Discount: {discount_percent}%")
        print(f"Discounted Price: Rs. {product['Price'] * (1 - discount_percent/100):.2f}")
        print(f"\nFinding top {max_customers} customers likely to buy...\n")
        
        # Get targeted customers using ML model
        campaign, summary = self.engine.create_promotion_campaign(
            product_id, 
            max_targets=max_customers,
            strategy='hybrid',
            optimize=False
        )
        
        if campaign is None or len(campaign) == 0:
            print("No suitable customers found!")
            return None
            
        # Add customer details
        campaign = campaign.merge(
            self.engine.preprocessor.customers[['CustomerID', 'Name', 'Age', 'Gender', 'Location', 'CustomerSegment']],
            on='CustomerID',
            how='left'
        )
        
        # Create campaign file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/campaign_{product_id}_{discount_percent}pct_{timestamp}.csv"
        
        # Select relevant columns for campaign (handle both optimized and non-optimized)
        base_cols = ['CustomerID', 'Name', 'Age', 'Gender', 'Location', 'CustomerSegment', 'purchase_probability']
        
        # Add optional columns if they exist
        if 'optimal_discount' in campaign.columns:
            base_cols.append('optimal_discount')
        if 'expected_roi' in campaign.columns:
            base_cols.append('expected_roi')
            
        campaign_output = campaign[base_cols].copy()
        
        campaign_output['ProductID'] = product_id
        campaign_output['ProductName'] = product['ProductName']
        campaign_output['OriginalPrice'] = product['Price']
        campaign_output['Discount%'] = discount_percent
        campaign_output['DiscountedPrice'] = product['Price'] * (1 - discount_percent/100)
        
        # Save to CSV
        campaign_output.to_csv(filename, index=False)
        
        # Calculate expected metrics
        avg_prob = campaign_output['purchase_probability'].mean()
        expected_conversions = avg_prob * len(campaign_output)
        expected_revenue = expected_conversions * campaign_output['DiscountedPrice'].iloc[0]
        discount_cost = expected_conversions * product['Price'] * (discount_percent/100)
        expected_profit = expected_revenue - discount_cost
        
        # Print summary
        print(f"{'='*70}")
        print(f" CAMPAIGN SUMMARY")
        print(f"{'='*70}")
        print(f"Your Discount: {discount_percent}%")
        print(f"Total Customers Targeted: {len(campaign_output)}")
        print(f"Average Purchase Probability: {avg_prob:.1%}")
        print(f"Expected Conversions: {expected_conversions:.1f} customers")
        print(f"Expected Revenue: Rs. {expected_revenue:,.2f}")
        print(f"Discount Cost: Rs. {discount_cost:,.2f}")
        print(f"Expected Profit: Rs. {expected_profit:,.2f}")
        print(f"\nCustomer Segments Distribution:")
        print(campaign_output['CustomerSegment'].value_counts())
        print(f"\n* Campaign saved to: {filename}")
        
        print(f"\n{'='*70}")
        print(f" TOP 10 CUSTOMERS TO TARGET")
        print(f"{'='*70}")
        print(campaign_output.head(10)[['CustomerID', 'Name', 'Location', 'CustomerSegment', 'purchase_probability']])
        
        return campaign_output
    
    def find_cross_sell_opportunities(self, product_id, top_n=5):
        """
        Find products frequently bought together (cross-selling)
        Example: Customers who buy bread often buy milk
        """
        print(f"\n{'='*70}")
        print(f" CROSS-SELL ANALYSIS")
        print(f"{'='*70}\n")
        
        product = self.engine.preprocessor.products[
            self.engine.preprocessor.products['ProductID'] == product_id
        ].iloc[0]
        
        print(f"Base Product: {product['ProductName']} ({product_id})")
        print(f"\nFinding products frequently bought together...\n")
        
        # Get customers who bought this product
        transactions = self.engine.preprocessor.transactions
        customers_who_bought = transactions[
            transactions['ProductID'] == product_id
        ]['CustomerID'].unique()
        
        print(f"Customers who bought {product['ProductName']}: {len(customers_who_bought)}")
        
        # Find what else these customers bought
        other_purchases = transactions[
            (transactions['CustomerID'].isin(customers_who_bought)) &
            (transactions['ProductID'] != product_id)
        ]
        
        # Count product co-occurrences
        product_counts = other_purchases.groupby('ProductID').agg({
            'TransactionID': 'count',
            'CustomerID': 'nunique'
        }).rename(columns={
            'TransactionID': 'total_purchases',
            'CustomerID': 'unique_customers'
        })
        
        product_counts['co_purchase_rate'] = product_counts['unique_customers'] / len(customers_who_bought)
        product_counts = product_counts.sort_values('co_purchase_rate', ascending=False).head(top_n)
        
        # Add product details
        product_counts = product_counts.merge(
            self.engine.preprocessor.products[['ProductID', 'ProductName', 'Category', 'Price']],
            on='ProductID',
            how='left'
        )
        
        print(f"\n{'='*70}")
        print(f" TOP {top_n} CROSS-SELL PRODUCTS")
        print(f"{'='*70}")
        print(f"{'Product':<40} {'Category':<15} {'Price':>10} {'Co-Buy Rate':>12}")
        print(f"{'-'*70}")
        
        for idx, row in product_counts.iterrows():
            print(f"{row['ProductName']:<40} {row['Category']:<15} Rs. {row['Price']:>7.2f} {row['co_purchase_rate']:>11.1%}")
        
        return product_counts
    
    def generate_cross_sell_campaign(self, base_product_id, cross_sell_product_id, discount_percent=10):
        """
        Generate campaign for cross-sell product targeting customers who bought base product
        Example: Send milk discount to bread buyers
        """
        print(f"\n{'='*70}")
        print(f" CROSS-SELL CAMPAIGN")
        print(f"{'='*70}\n")
        
        base_product = self.engine.preprocessor.products[
            self.engine.preprocessor.products['ProductID'] == base_product_id
        ].iloc[0]
        
        cross_product_match = self.engine.preprocessor.products[
            self.engine.preprocessor.products['ProductID'] == cross_sell_product_id
        ]
        
        if len(cross_product_match) == 0:
            print(f"Error: Product {cross_sell_product_id} not found!")
            return None
            
        cross_product = cross_product_match.iloc[0]
        
        print(f"Strategy: Send {cross_product['ProductName']} promotion")
        print(f"          to customers who bought {base_product['ProductName']}\n")
        
        # Get customers who bought base product but not cross-sell product yet
        transactions = self.engine.preprocessor.transactions
        
        base_buyers = set(transactions[
            transactions['ProductID'] == base_product_id
        ]['CustomerID'].unique())
        
        cross_buyers = set(transactions[
            transactions['ProductID'] == cross_sell_product_id
        ]['CustomerID'].unique())
        
        # Target customers who bought base but not cross-sell (upsell opportunity)
        target_customers = list(base_buyers - cross_buyers)
        
        print(f"Customers who bought {base_product['ProductName']}: {len(base_buyers)}")
        print(f"Customers who also bought {cross_product['ProductName']}: {len(cross_buyers)}")
        print(f"Upsell Opportunity: {len(target_customers)} customers\n")
        
        if len(target_customers) == 0:
            print("No upsell opportunity found!")
            return None
        
        # Get customer details
        customer_list = self.engine.preprocessor.customers[
            self.engine.preprocessor.customers['CustomerID'].isin(target_customers)
        ].copy()
        
        customer_list['BaseProduct'] = base_product['ProductName']
        customer_list['CrossSellProduct'] = cross_product['ProductName']
        customer_list['OriginalPrice'] = cross_product['Price']
        customer_list['Discount%'] = discount_percent
        customer_list['DiscountedPrice'] = cross_product['Price'] * (1 - discount_percent/100)
        customer_list['Strategy'] = 'Cross-Sell'
        
        # Save campaign
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/crosssell_{base_product_id}_to_{cross_sell_product_id}_{timestamp}.csv"
        customer_list.to_csv(filename, index=False)
        
        print(f"* Cross-sell campaign saved to: {filename}")
        print(f"\nSample customers to target:")
        print(customer_list.head(10)[['CustomerID', 'Name', 'Location', 'CustomerSegment']])
        
        return customer_list


def demo_use_cases():
    """Demonstrate practical use cases"""
    
    generator = CampaignGenerator()
    generator.load_models()
    
    print("\n" + "="*70)
    print(" PRACTICAL DEMONSTRATION - HOW TO USE THE MODELS")
    print("="*70)
    
    # Show available products
    products = generator.show_available_products()
    
    print("\n\n" + "="*70)
    print(" USE CASE 1: GENERATE PROMOTION FOR BREAD (10% OFF)")
    print("="*70)
    
    # Find a bread product
    bread = products[products['ProductName'].str.contains('Bread', case=False, na=False)].iloc[0]
    bread_id = bread['ProductID']
    
    # Generate campaign for bread
    bread_campaign = generator.generate_promotion_campaign(
        product_id=bread_id,
        discount_percent=10,
        max_customers=50
    )
    
    print("\n\n" + "="*70)
    print(" USE CASE 2: FIND CROSS-SELL OPPORTUNITIES FOR BREAD")
    print("="*70)
    
    # Find what products are bought with bread
    cross_sell_products = generator.find_cross_sell_opportunities(bread_id, top_n=5)
    
    if len(cross_sell_products) > 0:
        print("\n\n" + "="*70)
        print(" USE CASE 3: CROSS-SELL CAMPAIGN")
        print("="*70)
        
        # Get top cross-sell product - use reset_index to access ProductID
        cross_sell_products_reset = cross_sell_products.reset_index()
        top_cross_sell = cross_sell_products_reset.iloc[0]['ProductID']
        
        # Generate cross-sell campaign
        cross_campaign = generator.generate_cross_sell_campaign(
            base_product_id=bread_id,
            cross_sell_product_id=top_cross_sell,
            discount_percent=15
        )
    
    print("\n\n" + "="*70)
    print(" DEMONSTRATION COMPLETED!")
    print("="*70)
    print("\nAll campaign files saved in 'campaign_outputs/' folder")
    print("These CSV files contain the actual customer lists to send promotions to.")
    print("\nYou can now:")
    print("1. Show these CSV files in your presentation")
    print("2. Import them into email marketing systems")
    print("3. Use them for targeted SMS campaigns")
    print("4. Demonstrate the practical value of the ML models")


if __name__ == "__main__":
    demo_use_cases()
