"""
Data Validation Script
Checks if real e-commerce data is in correct format before training
"""

import pandas as pd
import os

def check_data_format():
    """Validate real data before model training"""
    
    print("="*70)
    print(" DATA VALIDATION FOR REAL E-COMMERCE DATA")
    print("="*70)
    
    data_dir = "data/raw"
    issues = []
    
    # Check 1: Files exist
    required_files = ['Customers.csv', 'Products.csv', 'Transactions.csv']
    print("\n1. Checking required files...")
    
    for file in required_files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"   * {file} found ({size:.1f} KB)")
        else:
            print(f"   X {file} MISSING!")
            issues.append(f"Missing file: {file}")
    
    if issues:
        print("\n! CRITICAL: Some files are missing!")
        return
    
    # Check 2: Customer data
    print("\n2. Validating Customers.csv...")
    customers = pd.read_csv(os.path.join(data_dir, 'Customers.csv'))
    print(f"   Total customers: {len(customers)}")
    
    required_cols = ['CustomerID']
    optional_cols = ['Email', 'Name', 'Age', 'Gender', 'Location']
    
    for col in required_cols:
        if col in customers.columns:
            print(f"   * {col} found")
        else:
            print(f"   X {col} MISSING (Required!)")
            issues.append(f"Customers.csv missing column: {col}")
    
    for col in optional_cols:
        if col in customers.columns:
            print(f"   * {col} found (optional)")
        else:
            print(f"   - {col} not found (optional, but recommended)")
    
    # Check 3: Product data
    print("\n3. Validating Products.csv...")
    products = pd.read_csv(os.path.join(data_dir, 'Products.csv'))
    print(f"   Total products: {len(products)}")
    
    required_cols = ['ProductID', 'Price']
    optional_cols = ['ProductName', 'Category', 'Brand']
    
    for col in required_cols:
        if col in products.columns:
            print(f"   * {col} found")
        else:
            print(f"   X {col} MISSING (Required!)")
            issues.append(f"Products.csv missing column: {col}")
    
    for col in optional_cols:
        if col in products.columns:
            print(f"   * {col} found (optional)")
        else:
            print(f"   - {col} not found (optional, but recommended)")
    
    # Check 4: Transaction data (MOST IMPORTANT!)
    print("\n4. Validating Transactions.csv...")
    transactions = pd.read_csv(os.path.join(data_dir, 'Transactions.csv'))
    print(f"   Total transactions: {len(transactions)}")
    
    required_cols = ['CustomerID', 'ProductID', 'TransactionDate']
    
    for col in required_cols:
        if col in transactions.columns:
            print(f"   * {col} found")
            
            # Check for valid data
            if col in ['CustomerID', 'ProductID']:
                unique_count = transactions[col].nunique()
                print(f"     -> {unique_count} unique values")
        else:
            print(f"   X {col} MISSING (Required!)")
            issues.append(f"Transactions.csv missing column: {col}")
    
    # Check 5: Data quality
    print("\n5. Data Quality Checks...")
    
    # Check for nulls
    customer_nulls = customers['CustomerID'].isnull().sum()
    product_nulls = products['ProductID'].isnull().sum()
    trans_nulls = transactions[['CustomerID', 'ProductID']].isnull().sum().sum()
    
    if customer_nulls == 0:
        print(f"   * No null CustomerIDs")
    else:
        print(f"   ! {customer_nulls} null CustomerIDs found!")
        issues.append(f"{customer_nulls} null CustomerIDs")
    
    if product_nulls == 0:
        print(f"   * No null ProductIDs")
    else:
        print(f"   ! {product_nulls} null ProductIDs found!")
        issues.append(f"{product_nulls} null ProductIDs")
    
    if trans_nulls == 0:
        print(f"   * No null IDs in Transactions")
    else:
        print(f"   ! {trans_nulls} null IDs in Transactions!")
        issues.append(f"{trans_nulls} null IDs in Transactions")
    
    # Check date format
    try:
        pd.to_datetime(transactions['TransactionDate'])
        print(f"   * TransactionDate format is valid")
    except:
        print(f"   ! TransactionDate format issue!")
        issues.append("Invalid date format in Transactions")
    
    # Check 6: Minimum data requirements
    print("\n6. Minimum Data Requirements...")
    
    min_customers = 100
    min_products = 20
    min_transactions = 1000
    
    if len(customers) >= min_customers:
        print(f"   * Sufficient customers ({len(customers)} >= {min_customers})")
    else:
        print(f"   ! Not enough customers ({len(customers)} < {min_customers})")
        print(f"      Recommendation: Need at least {min_customers} for good results")
    
    if len(products) >= min_products:
        print(f"   * Sufficient products ({len(products)} >= {min_products})")
    else:
        print(f"   ! Not enough products ({len(products)} < {min_products})")
        print(f"      Recommendation: Need at least {min_products} for good results")
    
    if len(transactions) >= min_transactions:
        print(f"   * Sufficient transactions ({len(transactions)} >= {min_transactions})")
    else:
        print(f"   ! Not enough transactions ({len(transactions)} < {min_transactions})")
        print(f"      Recommendation: Need at least {min_transactions} for good results")
    
    # Summary
    print("\n" + "="*70)
    if len(issues) == 0:
        print(" >> DATA VALIDATION PASSED!")
        print("="*70)
        print("\nYour data is ready for training!")
        print("\nNext steps:")
        print("1. Run: python data_analysis/preprocessing.py")
        print("2. Run: python models/purchase_prediction.py")
        print("3. Run: python models/collaborative_filtering.py")
        print("4. Test: python demo_campaign_generator.py")
    else:
        print(" >> DATA VALIDATION FAILED!")
        print("="*70)
        print(f"\nFound {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        print("\nPlease fix these issues before training the model.")
    print("="*70)


if __name__ == "__main__":
    try:
        check_data_format()
    except Exception as e:
        print(f"\nX Error: {e}")
        print("\nMake sure you have CSV files in data/raw/ folder:")
        print("  - Customers.csv")
        print("  - Products.csv")
        print("  - Transactions.csv")
