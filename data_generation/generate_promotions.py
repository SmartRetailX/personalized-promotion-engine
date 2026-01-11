"""
Generate synthetic promotions dataset
Creates historical promotions with various discount levels and durations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from config import *

np.random.seed(RANDOM_SEED)


def generate_promotions(products_df):
    """Generate promotions for various products"""
    
    print(f"Generating {NUM_PROMOTIONS} promotions...")
    
    promotions = []
    
    for i in range(NUM_PROMOTIONS):
        promotion_id = f'PROMO{i+1:05d}'
        
        # Select a random product
        product = products_df.sample(1).iloc[0]
        product_id = product['ProductID']
        
        # Random discount
        discount_pct = np.random.choice(DISCOUNT_PERCENTAGES)
        
        # Random start date within our time range
        start_date = START_DATE + timedelta(days=int(np.random.randint(0, DAYS_SPAN - 30)))
        
        # Random duration
        duration = int(np.random.choice(PROMOTION_DURATION_DAYS))
        end_date = start_date + timedelta(days=duration)
        
        # Ensure end date doesn't exceed our range
        if end_date > END_DATE:
            end_date = END_DATE
        
        promotion = {
            'PromotionID': promotion_id,
            'ProductID': product_id,
            'DiscountPercentage': discount_pct,
            'StartDate': start_date.strftime('%Y-%m-%d'),
            'EndDate': end_date.strftime('%Y-%m-%d'),
            'PromotionType': np.random.choice(['Flash Sale', 'Weekly Deal', 'Clearance', 'Bundle Offer']),
            'TargetedPromotion': np.random.choice([True, False], p=[0.3, 0.7])  # 30% are targeted
        }
        
        promotions.append(promotion)
    
    df = pd.DataFrame(promotions)
    return df


def main():
    """Main function to generate and save promotions dataset"""
    
    # Load products to reference
    products_path = os.path.join(OUTPUT_DIR, 'Products.csv')
    if not os.path.exists(products_path):
        print("Error: Products.csv not found. Please run generate_products.py first.")
        return None
    
    products_df = pd.read_csv(products_path)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate promotions
    promotions_df = generate_promotions(products_df)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'Promotions.csv')
    promotions_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Promotions dataset saved to: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total Promotions: {len(promotions_df)}")
    print(f"\nDiscount Distribution:")
    print(promotions_df['DiscountPercentage'].value_counts().sort_index())
    print(f"\nPromotion Type Distribution:")
    print(promotions_df['PromotionType'].value_counts())
    print(f"\nTargeted vs Broadcast:")
    print(promotions_df['TargetedPromotion'].value_counts())
    
    return promotions_df


if __name__ == "__main__":
    promotions_df = main()
