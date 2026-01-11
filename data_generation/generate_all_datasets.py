"""
Master script to generate all datasets in correct order
Run this to create the complete synthetic dataset
"""

import os
import sys
from datetime import datetime

# Import all generators
import generate_customers
import generate_products
import generate_stores
import generate_promotions
import generate_transactions

from config import OUTPUT_DIR


def main():
    """Generate all datasets"""
    
    print("=" * 70)
    print(" PERSONALIZED PROMOTION ENGINE - DATASET GENERATION")
    print("=" * 70)
    print(f"\nGeneration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Generate Customers
        print("\n" + "=" * 70)
        print("STEP 1/5: Generating Customers")
        print("=" * 70)
        customers_df = generate_customers.main()
        
        # Step 2: Generate Products
        print("\n" + "=" * 70)
        print("STEP 2/5: Generating Products")
        print("=" * 70)
        products_df = generate_products.main()
        
        # Step 3: Generate Stores
        print("\n" + "=" * 70)
        print("STEP 3/5: Generating Stores")
        print("=" * 70)
        stores_df = generate_stores.main()
        
        # Step 4: Generate Promotions
        print("\n" + "=" * 70)
        print("STEP 4/5: Generating Promotions")
        print("=" * 70)
        promotions_df = generate_promotions.main()
        
        # Step 5: Generate Transactions (most time-consuming)
        print("\n" + "=" * 70)
        print("STEP 5/5: Generating Transactions")
        print("=" * 70)
        transactions_df = generate_transactions.main()
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print(" DATASET GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nGeneration completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time taken: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        print("\n" + "-" * 70)
        print("DATASET SUMMARY")
        print("-" * 70)
        print(f"  Customers:    {len(customers_df):,}")
        print(f"  Products:     {len(products_df):,}")
        print(f"  Stores:       {len(stores_df):,}")
        print(f"  Promotions:   {len(promotions_df):,}")
        print(f"  Transactions: {len(transactions_df):,}")
        
        print("\n" + "-" * 70)
        print("NEXT STEPS")
        print("-" * 70)
        print("1. Run exploratory data analysis:")
        print("   python data_analysis/eda.py")
        print("\n2. Or open the Jupyter notebook:")
        print("   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb")
        print("\n3. Then proceed to model training:")
        print("   python models/train_all_models.py")
        
        print("\n" + "=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during dataset generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
