"""
Configuration file for synthetic data generation
Adjust these parameters to control dataset size and characteristics
"""

import random
from datetime import datetime, timedelta

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Dataset Size Parameters
NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 250
NUM_TRANSACTIONS = 50000
NUM_STORES = 10
NUM_PROMOTIONS = 200

# Time Parameters
START_DATE = datetime(2024, 7, 1)  # 18 months ago from now
END_DATE = datetime(2025, 12, 31)
DAYS_SPAN = (END_DATE - START_DATE).days

# Customer Demographics
AGE_RANGE = (18, 75)
GENDERS = ['Male', 'Female', 'Other']
GENDER_DISTRIBUTION = [0.48, 0.48, 0.04]

# Sri Lankan Cities (for realistic location data)
SRI_LANKAN_CITIES = [
    'Colombo', 'Kandy', 'Galle', 'Jaffna', 'Negombo',
    'Kurunegala', 'Anuradhapura', 'Trincomalee', 'Batticaloa',
    'Matara', 'Ratnapura', 'Badulla', 'Nuwara Eliya', 'Kalutara',
    'Gampaha', 'Moratuwa', 'Kilinochchi', 'Vavuniya'
]

# Product Categories (expanded for better analysis)
PRODUCT_CATEGORIES = {
    'Bakery': {
        'products': ['White Bread', 'Brown Bread', 'Buns', 'Croissants', 'Pastries', 'Cakes', 'Donuts'],
        'price_range': (80, 500),
        'purchase_frequency': 'high'  # bought frequently
    },
    'Dairy': {
        'products': ['Milk', 'Yogurt', 'Cheese', 'Butter', 'Cream', 'Ice Cream', 'Curd'],
        'price_range': (150, 800),
        'purchase_frequency': 'high'
    },
    'Beverages': {
        'products': ['Soft Drinks', 'Juice', 'Water', 'Tea', 'Coffee', 'Energy Drinks', 'Cordials'],
        'price_range': (100, 1200),
        'purchase_frequency': 'high'
    },
    'Fruits': {
        'products': ['Apples', 'Bananas', 'Oranges', 'Grapes', 'Mangoes', 'Pineapples', 'Papaya', 'Watermelon'],
        'price_range': (200, 1000),
        'purchase_frequency': 'medium'
    },
    'Vegetables': {
        'products': ['Tomatoes', 'Onions', 'Potatoes', 'Carrots', 'Cabbage', 'Beans', 'Leeks', 'Brinjal'],
        'price_range': (150, 800),
        'purchase_frequency': 'high'
    },
    'Meat': {
        'products': ['Chicken', 'Beef', 'Pork', 'Fish', 'Prawns', 'Mutton', 'Sausages'],
        'price_range': (500, 3000),
        'purchase_frequency': 'medium'
    },
    'Snacks': {
        'products': ['Chips', 'Biscuits', 'Chocolates', 'Nuts', 'Crackers', 'Wafers', 'Candy'],
        'price_range': (100, 800),
        'purchase_frequency': 'medium'
    },
    'Household': {
        'products': ['Detergent', 'Soap', 'Shampoo', 'Toothpaste', 'Tissues', 'Toilet Paper', 'Cleaners'],
        'price_range': (200, 1500),
        'purchase_frequency': 'low'
    },
    'Rice & Grains': {
        'products': ['White Rice', 'Red Rice', 'Pasta', 'Noodles', 'Flour', 'Oats', 'Cereals'],
        'price_range': (300, 2500),
        'purchase_frequency': 'low'
    },
    'Spices': {
        'products': ['Chili Powder', 'Curry Powder', 'Turmeric', 'Cinnamon', 'Pepper', 'Cardamom', 'Salt'],
        'price_range': (150, 1200),
        'purchase_frequency': 'low'
    },
    'Frozen Foods': {
        'products': ['Frozen Vegetables', 'Frozen Fries', 'Frozen Fish', 'Frozen Chicken', 'Frozen Pizza'],
        'price_range': (400, 2000),
        'purchase_frequency': 'low'
    },
    'Canned Foods': {
        'products': ['Canned Fish', 'Canned Beans', 'Canned Fruits', 'Canned Vegetables', 'Sauces'],
        'price_range': (200, 800),
        'purchase_frequency': 'low'
    },
    'Baby Products': {
        'products': ['Diapers', 'Baby Food', 'Baby Powder', 'Baby Soap', 'Wet Wipes'],
        'price_range': (300, 2500),
        'purchase_frequency': 'medium'
    },
    'Personal Care': {
        'products': ['Face Wash', 'Moisturizer', 'Deodorant', 'Perfume', 'Hair Oil', 'Body Lotion'],
        'price_range': (250, 3000),
        'purchase_frequency': 'low'
    },
    'Breakfast Items': {
        'products': ['Jam', 'Honey', 'Peanut Butter', 'Spreads', 'Breakfast Cereals'],
        'price_range': (300, 1500),
        'purchase_frequency': 'low'
    }
}

# Popular Sri Lankan brands (realistic)
BRANDS = [
    'Keells', 'Anchor', 'Maliban', 'Munchee', 'Elephant House',
    'Kotmale', 'Coca-Cola', 'Nestlé', 'Unilever', 'Prima',
    'MD', 'CBL', 'Raigam', 'Tiara', 'Richlife',
    'Generic', 'Store Brand', 'Imported'
]

# Store Types
STORE_TYPES = ['Supermarket', 'Hypermarket', 'Express Store']

# Customer Behavior Patterns (for realistic transaction generation)
CUSTOMER_SEGMENTS = {
    'frequent_shoppers': {
        'percentage': 0.20,  # 20% of customers
        'avg_transactions_per_month': 12,
        'avg_basket_size': 8,
        'price_sensitivity': 'high'  # respond well to promotions
    },
    'regular_shoppers': {
        'percentage': 0.45,
        'avg_transactions_per_month': 6,
        'avg_basket_size': 5,
        'price_sensitivity': 'medium'
    },
    'occasional_shoppers': {
        'percentage': 0.25,
        'avg_transactions_per_month': 3,
        'avg_basket_size': 3,
        'price_sensitivity': 'low'
    },
    'rare_shoppers': {
        'percentage': 0.10,
        'avg_transactions_per_month': 1,
        'avg_basket_size': 2,
        'price_sensitivity': 'low'
    }
}

# Promotion Parameters
DISCOUNT_PERCENTAGES = [5, 10, 15, 20, 25, 30, 40, 50]
PROMOTION_DURATION_DAYS = [3, 7, 14, 30]  # Common promotion durations

# Transaction Parameters
MIN_QUANTITY = 1
MAX_QUANTITY = 10
PROMOTION_RESPONSE_RATE = 0.25  # 25% of targeted customers respond to promotions

# Seasonality and Trends (for realistic patterns)
# Month -> category purchase multiplier
SEASONAL_PATTERNS = {
    'December': {'Bakery': 1.5, 'Beverages': 1.3, 'Snacks': 1.4},  # Festive season
    'April': {'Beverages': 1.3, 'Fruits': 1.2},  # New Year
    'July': {'Household': 1.2},  # Mid-year
}

# Product Affinity (products often bought together)
PRODUCT_AFFINITIES = {
    'Bread': ['Butter', 'Jam', 'Cheese', 'Milk'],
    'Tea': ['Biscuits', 'Milk', 'Sugar'],
    'Coffee': ['Milk', 'Cream', 'Biscuits'],
    'Rice': ['Curry Powder', 'Onions', 'Chicken', 'Fish'],
    'Pasta': ['Cheese', 'Tomatoes', 'Sauces'],
    'Chips': ['Soft Drinks', 'Juice'],
    'Chicken': ['Rice', 'Vegetables', 'Spices'],
}

# Output Paths
OUTPUT_DIR = 'f:\\.1 Research\\Personalized Promotion Engine\\data\\raw'
PROCESSED_DIR = 'f:\\.1 Research\\Personalized Promotion Engine\\data\\processed'
