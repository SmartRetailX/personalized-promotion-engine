"""
Simple Interactive Script - Generate Promotion Campaigns
Use this to demonstrate your research in presentations
"""

import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from demo_campaign_generator import CampaignGenerator

def main():
    print("\n" + "="*70)
    print(" PERSONALIZED PROMOTION CAMPAIGN GENERATOR")
    print(" Research Project - SLIIT")
    print("="*70)
    
    print("\nInitializing AI models...")
    generator = CampaignGenerator()
    generator.load_models()
    
    while True:
        print("\n" + "="*70)
        print(" MAIN MENU")
        print("="*70)
        print("1. Show available products")
        print("2. Generate promotion campaign for a product")
        print("3. Find cross-sell opportunities")
        print("4. Generate cross-sell campaign")
        print("5. View generated campaigns")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            category = input("\nFilter by category (press Enter for all): ").strip()
            generator.show_available_products(category if category else None)
            
        elif choice == '2':
            product_id = input("\nEnter Product ID (e.g., PROD00001): ").strip().upper()
            discount = input("Enter discount percentage (default 10): ").strip()
            max_customers = input("Enter max customers to target (default 50): ").strip()
            
            discount = int(discount) if discount else 10
            max_customers = int(max_customers) if max_customers else 50
            
            generator.generate_promotion_campaign(product_id, discount, max_customers)
            
        elif choice == '3':
            product_id = input("\nEnter Product ID to analyze (e.g., PROD00001): ").strip().upper()
            generator.find_cross_sell_opportunities(product_id, top_n=5)
            
        elif choice == '4':
            base_id = input("\nEnter base Product ID (e.g., PROD00001): ").strip().upper()
            cross_id = input("Enter cross-sell Product ID (e.g., PROD00015): ").strip().upper()
            discount = input("Enter discount percentage (default 10): ").strip()
            
            discount = int(discount) if discount else 10
            generator.generate_cross_sell_campaign(base_id, cross_id, discount)
            
        elif choice == '5':
            import glob
            campaigns = glob.glob("campaign_outputs/*.csv")
            print(f"\n{'='*70}")
            print(f" GENERATED CAMPAIGNS ({len(campaigns)} files)")
            print(f"{'='*70}")
            for camp in campaigns:
                size = os.path.getsize(camp)
                print(f"{os.path.basename(camp):<60} {size:>8} bytes")
            
        elif choice == '6':
            print("\nThank you for using the Promotion Engine!")
            break
        else:
            print("\nInvalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
