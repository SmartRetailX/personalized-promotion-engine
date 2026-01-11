"""
Generate synthetic transactions dataset
Creates realistic purchase history with customer behavior patterns
This is the most complex generator as it needs to simulate:
- Customer shopping patterns (frequency, basket size)
- Product affinities (items bought together)
- Seasonality effects
- Promotion responses
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from config import *
from tqdm import tqdm

np.random.seed(RANDOM_SEED)


def get_customer_preferences(customer_segment):
    """Generate customer preferences for product categories"""
    preferences = {}
    
    for category in PRODUCT_CATEGORIES.keys():
        # Different segments prefer different categories
        if customer_segment == 'frequent_shoppers':
            # Frequent shoppers buy more essentials
            if category in ['Bakery', 'Dairy', 'Vegetables', 'Beverages']:
                preferences[category] = np.random.uniform(0.7, 1.0)
            else:
                preferences[category] = np.random.uniform(0.3, 0.7)
        else:
            preferences[category] = np.random.uniform(0.2, 0.8)
    
    return preferences


def get_seasonal_multiplier(date, category):
    """Get seasonal multiplier for a category on a specific date"""
    month_name = date.strftime('%B')
    
    if month_name in SEASONAL_PATTERNS:
        if category in SEASONAL_PATTERNS[month_name]:
            return SEASONAL_PATTERNS[month_name][category]
    
    return 1.0


def generate_transaction_date(customer_segment, start_date, end_date, existing_dates):
    """Generate a realistic transaction date based on customer segment"""
    
    segment_info = CUSTOMER_SEGMENTS[customer_segment]
    avg_days_between = 30 / segment_info['avg_transactions_per_month']
    
    if existing_dates:
        # Generate next transaction after last one
        last_date = max(existing_dates)
        days_to_add = int(np.random.exponential(avg_days_between))
        new_date = last_date + timedelta(days=days_to_add)
        
        if new_date > end_date:
            return None
    else:
        # First transaction - random date in range
        days_from_start = np.random.randint(0, (end_date - start_date).days)
        new_date = start_date + timedelta(days=days_from_start)
    
    return new_date


def select_products_for_basket(products_df, customer_preferences, basket_size, transaction_date):
    """Select products for a shopping basket based on preferences and affinities"""
    
    selected_products = []
    
    for _ in range(basket_size):
        # Select category based on preferences and seasonality
        categories = list(customer_preferences.keys())
        weights = []
        
        for cat in categories:
            pref = customer_preferences[cat]
            seasonal = get_seasonal_multiplier(transaction_date, cat)
            weights.append(pref * seasonal)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        selected_category = np.random.choice(categories, p=weights)
        
        # Select product from category
        category_products = products_df[products_df['Category'] == selected_category]
        if len(category_products) > 0:
            product = category_products.sample(1).iloc[0]
            selected_products.append(product)
    
    return selected_products


def generate_transactions(customers_df, products_df, promotions_df, stores_df):
    """Generate realistic transactions"""
    
    print(f"Generating {NUM_TRANSACTIONS} transactions...")
    
    transactions = []
    transaction_id = 1
    
    # Pre-calculate customer preferences
    customer_prefs = {}
    for _, customer in customers_df.iterrows():
        customer_prefs[customer['CustomerID']] = get_customer_preferences(customer['CustomerSegment'])
    
    # Track customer transaction dates
    customer_transaction_dates = {cust_id: [] for cust_id in customers_df['CustomerID']}
    
    # Convert promotions to a more usable format
    promotions_df['StartDate'] = pd.to_datetime(promotions_df['StartDate'])
    promotions_df['EndDate'] = pd.to_datetime(promotions_df['EndDate'])
    
    with tqdm(total=NUM_TRANSACTIONS) as pbar:
        while transaction_id <= NUM_TRANSACTIONS:
            # Select random customer
            customer = customers_df.sample(1).iloc[0]
            customer_id = customer['CustomerID']
            customer_segment = customer['CustomerSegment']
            
            # Generate transaction date
            trans_date = generate_transaction_date(
                customer_segment,
                START_DATE,
                END_DATE,
                customer_transaction_dates[customer_id]
            )
            
            if trans_date is None:
                continue
            
            customer_transaction_dates[customer_id].append(trans_date)
            
            # Determine basket size
            segment_info = CUSTOMER_SEGMENTS[customer_segment]
            basket_size = max(1, int(np.random.poisson(segment_info['avg_basket_size'])))
            
            # Select products for basket
            basket_products = select_products_for_basket(
                products_df,
                customer_prefs[customer_id],
                basket_size,
                trans_date
            )
            
            # Select store (prefer stores in customer's location)
            if np.random.random() < 0.7:  # 70% shop in same city
                city_stores = stores_df[stores_df['City'] == customer['Location']]
                if len(city_stores) > 0:
                    store = city_stores.sample(1).iloc[0]
                else:
                    store = stores_df.sample(1).iloc[0]
            else:
                store = stores_df.sample(1).iloc[0]
            
            # Create transaction for each product in basket
            for product in basket_products:
                # Check if there's an active promotion
                active_promos = promotions_df[
                    (promotions_df['ProductID'] == product['ProductID']) &
                    (promotions_df['StartDate'] <= trans_date) &
                    (promotions_df['EndDate'] >= trans_date)
                ]
                
                if len(active_promos) > 0:
                    promo = active_promos.sample(1).iloc[0]
                    promotion_id = promo['PromotionID']
                    discount_pct = promo['DiscountPercentage']
                    
                    # Customer responds to promotion based on price sensitivity
                    price_sensitivity = CUSTOMER_SEGMENTS[customer_segment]['price_sensitivity']
                    if price_sensitivity == 'high':
                        responds = np.random.random() < 0.7
                    elif price_sensitivity == 'medium':
                        responds = np.random.random() < 0.5
                    else:
                        responds = np.random.random() < 0.3
                    
                    if not responds:
                        promotion_id = None
                        discount_pct = 0
                else:
                    promotion_id = None
                    discount_pct = 0
                
                # Quantity
                quantity = np.random.randint(MIN_QUANTITY, min(MAX_QUANTITY, 5) + 1)
                
                # Pricing
                unit_price = product['Price']
                discount_amount = (unit_price * quantity * discount_pct / 100) if discount_pct > 0 else 0
                
                transaction = {
                    'TransactionID': f'TRANS{transaction_id:06d}',
                    'CustomerID': customer_id,
                    'ProductID': product['ProductID'],
                    'Quantity': quantity,
                    'UnitPrice': unit_price,
                    'TransactionDate': trans_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'DiscountedAmount': round(discount_amount, 2),
                    'PromotionID': promotion_id if promotion_id else 'None',
                    'StoreID': store['StoreID'],
                    'TotalAmount': round(unit_price * quantity - discount_amount, 2)
                }
                
                transactions.append(transaction)
                transaction_id += 1
                pbar.update(1)
                
                if transaction_id > NUM_TRANSACTIONS:
                    break
            
            if transaction_id > NUM_TRANSACTIONS:
                break
    
    df = pd.DataFrame(transactions)
    return df


def main():
    """Main function to generate and save transactions dataset"""
    
    # Load prerequisite datasets
    customers_path = os.path.join(OUTPUT_DIR, 'Customers.csv')
    products_path = os.path.join(OUTPUT_DIR, 'Products.csv')
    promotions_path = os.path.join(OUTPUT_DIR, 'Promotions.csv')
    stores_path = os.path.join(OUTPUT_DIR, 'Stores.csv')
    
    required_files = [
        (customers_path, "Customers.csv"),
        (products_path, "Products.csv"),
        (promotions_path, "Promotions.csv"),
        (stores_path, "Stores.csv")
    ]
    
    for file_path, file_name in required_files:
        if not os.path.exists(file_path):
            print(f"Error: {file_name} not found. Please run the respective generator first.")
            return None
    
    # Load datasets
    customers_df = pd.read_csv(customers_path)
    products_df = pd.read_csv(products_path)
    promotions_df = pd.read_csv(promotions_path)
    stores_df = pd.read_csv(stores_path)
    
    # Generate transactions
    transactions_df = generate_transactions(customers_df, products_df, promotions_df, stores_df)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'Transactions.csv')
    transactions_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Transactions dataset saved to: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total Transactions: {len(transactions_df)}")
    print(f"\nTransaction Amount Statistics:")
    print(transactions_df['TotalAmount'].describe())
    print(f"\nTransactions with Promotions:")
    print(f"  {len(transactions_df[transactions_df['PromotionID'] != 'None'])} ({len(transactions_df[transactions_df['PromotionID'] != 'None'])/len(transactions_df)*100:.1f}%)")
    print(f"\nTotal Revenue: Rs. {transactions_df['TotalAmount'].sum():,.2f}")
    print(f"\nTotal Discounts: Rs. {transactions_df['DiscountedAmount'].sum():,.2f}")
    
    return transactions_df


if __name__ == "__main__":
    transactions_df = main()
