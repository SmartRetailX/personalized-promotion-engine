import pandas as pd

# 1. Load the datasets
transactions = pd.read_csv('Transactions.csv')
products = pd.read_csv('Products.csv')

# Merge transactions with product details to see names and categories
df = pd.merge(transactions, products, on='ProductID')

# --- LOYALTY CHECK ---
print("--- LOYALTY CHECK (BREAD) ---")

# Filter for Bakery items (assuming Bread is in 'Bakery')
# If you have a specific 'ProductName' column, you can use: df[df['ProductName'].str.contains('Bread')]
bread_df = df[df['Category'] == 'Bakery']

# Count transactions per customer
loyalty_ranking = bread_df.groupby('CustomerID').size().reset_index(name='Purchase_Count')
loyalty_ranking = loyalty_ranking.sort_values(by='Purchase_Count', ascending=False)

print(f"Total customers who bought bread: {len(loyalty_ranking)}")
print("Top 5 'Bread Lovers' (Target for 10% discount):")
print(loyalty_ranking.head(5))


# --- AFFINITY CHECK ---
print("\n--- AFFINITY CHECK (BREAD + MILK) ---")

# Define a 'Basket' as (CustomerID + TransactionDate)
df['BasketID'] = df['CustomerID'].astype(str) + "_" + df['TransactionDate'].astype(str)

# Identify Baskets that contain Bread (Bakery)
bread_baskets = df[df['Category'] == 'Bakery']['BasketID'].unique()

# Filter the main dataframe to only include these specific baskets
items_in_bread_baskets = df[df['BasketID'].isin(bread_baskets)]

# Check how many of these baskets also contain Milk (Dairy)
milk_and_bread_baskets = items_in_bread_baskets[items_in_bread_baskets['Category'] == 'Dairy']['BasketID'].nunique()

total_bread_baskets = len(bread_baskets)
affinity_percent = (milk_and_bread_baskets / total_bread_baskets) * 100

print(f"Total baskets with Bread: {total_bread_baskets}")
print(f"Baskets containing both Bread AND Milk: {milk_and_bread_baskets}")
print(f"Support/Affinity: {affinity_percent:.2f}% of bread buyers also bought milk.")