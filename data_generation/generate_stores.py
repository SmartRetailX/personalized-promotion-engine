"""
Generate synthetic stores dataset
Creates store locations across Sri Lankan cities
"""

import pandas as pd
import numpy as np
import os
from config import *

np.random.seed(RANDOM_SEED)


def generate_stores():
    """Generate stores with locations and types"""
    
    print(f"Generating {NUM_STORES} stores...")
    
    stores = []
    
    for i in range(NUM_STORES):
        store_id = f'STORE{i+1:03d}'
        city = np.random.choice(SRI_LANKAN_CITIES)
        
        # Larger cities more likely to have hypermarkets
        if city in ['Colombo', 'Kandy', 'Galle', 'Negombo']:
            store_type = np.random.choice(STORE_TYPES, p=[0.3, 0.5, 0.2])
        else:
            store_type = np.random.choice(STORE_TYPES, p=[0.5, 0.2, 0.3])
        
        store = {
            'StoreID': store_id,
            'City': city,
            'StoreType': store_type,
            'OpeningDate': (START_DATE - timedelta(days=np.random.randint(365, 3650))).strftime('%Y-%m-%d')
        }
        
        stores.append(store)
    
    df = pd.DataFrame(stores)
    return df


def main():
    """Main function to generate and save stores dataset"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate stores
    stores_df = generate_stores()
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'Stores.csv')
    stores_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Stores dataset saved to: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total Stores: {len(stores_df)}")
    print(f"\nStore Type Distribution:")
    print(stores_df['StoreType'].value_counts())
    print(f"\nCity Distribution:")
    print(stores_df['City'].value_counts())
    
    return stores_df


if __name__ == "__main__":
    stores_df = main()
