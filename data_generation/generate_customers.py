"""
Generate synthetic customers dataset
Creates realistic customer demographics and behavioral segments
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import os
import sys

# Import configuration
from config import *

fake = Faker()
np.random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)


def generate_customers():
    """Generate customers with demographics and behavioral segments"""
    
    print(f"Generating {NUM_CUSTOMERS} customers...")
    
    customers = []
    customer_id = 1
    
    for segment_name, segment_info in CUSTOMER_SEGMENTS.items():
        num_in_segment = int(NUM_CUSTOMERS * segment_info['percentage'])
        
        for _ in range(num_in_segment):
            # Registration date (spread over past 2-3 years)
            days_ago = np.random.randint(60, 1095)  # 2 months to 3 years ago
            registration_date = END_DATE - timedelta(days=days_ago)
            
            customer = {
                'CustomerID': f'CUST{customer_id:05d}',
                'Name': fake.name(),
                'Age': np.random.randint(AGE_RANGE[0], AGE_RANGE[1] + 1),
                'Gender': np.random.choice(GENDERS, p=GENDER_DISTRIBUTION),
                'Location': np.random.choice(SRI_LANKAN_CITIES),
                'RegistrationDate': registration_date.strftime('%Y-%m-%d'),
                'CustomerSegment': segment_name,  # Added for analysis
                'Email': fake.email(),
                'PhoneNumber': f'07{np.random.randint(10000000, 99999999)}'  # Sri Lankan mobile format
            }
            
            customers.append(customer)
            customer_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(customers)
    
    # Reorder columns
    df = df[['CustomerID', 'Name', 'Age', 'Gender', 'Location', 'RegistrationDate', 
             'CustomerSegment', 'Email', 'PhoneNumber']]
    
    return df


def main():
    """Main function to generate and save customers dataset"""
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate customers
    customers_df = generate_customers()
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'Customers.csv')
    customers_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Customers dataset saved to: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total Customers: {len(customers_df)}")
    print(f"\nCustomer Segments:")
    print(customers_df['CustomerSegment'].value_counts())
    print(f"\nAge Distribution:")
    print(customers_df['Age'].describe())
    print(f"\nGender Distribution:")
    print(customers_df['Gender'].value_counts())
    print(f"\nLocation Distribution (Top 10):")
    print(customers_df['Location'].value_counts().head(10))
    
    return customers_df


if __name__ == "__main__":
    customers_df = main()
