"""
Generate synthetic products dataset
Creates products across multiple categories with realistic pricing
"""

import pandas as pd
import numpy as np
import os
from config import *

np.random.seed(RANDOM_SEED)


def generate_products():
    """Generate products with categories, brands, and pricing"""
    
    print(f"Generating {NUM_PRODUCTS} products across {len(PRODUCT_CATEGORIES)} categories...")
    
    products = []
    product_id = 1
    
    # Calculate products per category
    products_per_category = NUM_PRODUCTS // len(PRODUCT_CATEGORIES)
    
    for category, category_info in PRODUCT_CATEGORIES.items():
        base_products = category_info['products']
        price_range = category_info['price_range']
        frequency = category_info['purchase_frequency']
        
        # Generate variations for each base product
        for base_product in base_products:
            # Create variations with different brands
            num_variations = max(1, products_per_category // len(base_products))
            
            for i in range(num_variations):
                brand = np.random.choice(BRANDS)
                
                # Add size/variant info for some products
                variants = ['', 'Small', 'Medium', 'Large', '500g', '1kg', '2L', '500ml']
                variant = np.random.choice(variants, p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
                
                product_name = f"{brand} {base_product} {variant}".strip()
                
                # Generate price with some variation
                base_price = np.random.uniform(price_range[0], price_range[1])
                price = round(base_price / 10) * 10  # Round to nearest 10
                
                product = {
                    'ProductID': f'PROD{product_id:05d}',
                    'ProductName': product_name,
                    'Category': category,
                    'Brand': brand,
                    'Price': price,
                    'BaseProduct': base_product,  # For analysis
                    'PurchaseFrequency': frequency  # For pattern generation
                }
                
                products.append(product)
                product_id += 1
                
                if product_id > NUM_PRODUCTS:
                    break
            
            if product_id > NUM_PRODUCTS:
                break
        
        if product_id > NUM_PRODUCTS:
            break
    
    # Create DataFrame
    df = pd.DataFrame(products)
    
    # Ensure we have exactly NUM_PRODUCTS
    df = df.head(NUM_PRODUCTS)
    
    return df


def main():
    """Main function to generate and save products dataset"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate products
    products_df = generate_products()
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'Products.csv')
    products_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Products dataset saved to: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total Products: {len(products_df)}")
    print(f"\nCategory Distribution:")
    print(products_df['Category'].value_counts())
    print(f"\nPrice Statistics:")
    print(products_df['Price'].describe())
    print(f"\nBrand Distribution (Top 10):")
    print(products_df['Brand'].value_counts().head(10))
    
    return products_df


if __name__ == "__main__":
    products_df = main()
