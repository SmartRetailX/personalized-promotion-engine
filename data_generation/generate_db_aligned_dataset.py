"""
DB-Aligned Synthetic Dataset Generator
Generates data that exactly matches the Prisma schema in e_commerce_platform_schema.prizma

Schema key points:
- user: id (string, not uuid), name, email, age?, gender?, City?, customerSegment?, role
- Product: productId (uuid), sku, name, categoryId (uuid), price, stockQuantity, brand, purchaseFrequency
- Category: id (uuid), name
- Promotion: promotionId (uuid), productId (uuid), discountPercentage, startDate, endDate,
             promotionType, isTargettedPromotion, productScope, status
- Transaction: transactionId (uuid), orderId (uuid), invoiceNo, customerId (string),
               productId (uuid), quantity, unitPrice, totalAmount, transactionDate,
               discountAmount, promotionId? (uuid)
- Order: id (uuid), orderNumber, userId (string), status, subtotal, discount, tax, total

Usage:
    python data_generation/generate_db_aligned_dataset.py
    
Outputs: data/raw_db/ with CSV files named after DB tables
"""

import uuid
import random
import csv
import os
from datetime import datetime, timedelta
from decimal import Decimal

# ─────────────────────────── Config ───────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

NUM_USERS = 1000
NUM_PRODUCTS = 250
NUM_PROMOTIONS = 200
NUM_ORDERS = 5000          # parent orders
NUM_TRANSACTIONS = 50000   # line items across all orders

START_DATE = datetime(2024, 7, 1)
END_DATE = datetime(2025, 12, 31)
DAYS_SPAN = (END_DATE - START_DATE).days

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw_db")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────── Helpers ──────────────────────────
def new_uuid():
    return str(uuid.uuid4())

def rand_date(start=START_DATE, end=END_DATE):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def fmt_ts(dt: datetime) -> str:
    """ISO timestamp with timezone offset (UTC)"""
    return dt.strftime("%Y-%m-%d %H:%M:%S+00")

# ─────────────────────────── Categories ───────────────────────
CATEGORIES_DEF = {
    "Bakery":           {"price_range": (80, 500),   "purchase_frequency": "high"},
    "Dairy":            {"price_range": (150, 800),  "purchase_frequency": "high"},
    "Beverages":        {"price_range": (100, 1200), "purchase_frequency": "high"},
    "Fruits":           {"price_range": (200, 1000), "purchase_frequency": "medium"},
    "Vegetables":       {"price_range": (150, 800),  "purchase_frequency": "high"},
    "Meat":             {"price_range": (500, 3000), "purchase_frequency": "medium"},
    "Snacks":           {"price_range": (100, 800),  "purchase_frequency": "medium"},
    "Household":        {"price_range": (200, 1500), "purchase_frequency": "low"},
    "Rice & Grains":    {"price_range": (300, 2500), "purchase_frequency": "low"},
    "Spices":           {"price_range": (150, 1200), "purchase_frequency": "low"},
    "Frozen Foods":     {"price_range": (400, 2000), "purchase_frequency": "low"},
    "Canned Foods":     {"price_range": (200, 800),  "purchase_frequency": "low"},
    "Baby Products":    {"price_range": (300, 2500), "purchase_frequency": "medium"},
    "Personal Care":    {"price_range": (250, 3000), "purchase_frequency": "low"},
    "Breakfast Items":  {"price_range": (300, 1500), "purchase_frequency": "low"},
}

PRODUCT_NAMES_BY_CAT = {
    "Bakery":          ["White Bread", "Brown Bread", "Buns", "Croissants", "Pastries", "Cakes", "Donuts", "Roti", "Scones", "Muffins"],
    "Dairy":           ["Milk", "Yogurt", "Cheese", "Butter", "Cream", "Ice Cream", "Curd", "Ghee", "Condensed Milk", "Kefir"],
    "Beverages":       ["Soft Drinks", "Juice", "Water", "Tea", "Coffee", "Energy Drinks", "Cordials", "Coconut Water", "Herbal Tea", "Iced Tea"],
    "Fruits":          ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes", "Pineapples", "Papaya", "Watermelon", "Strawberries", "Guava"],
    "Vegetables":      ["Tomatoes", "Onions", "Potatoes", "Carrots", "Cabbage", "Beans", "Leeks", "Brinjal", "Okra", "Pumpkin"],
    "Meat":            ["Chicken", "Beef", "Pork", "Fish", "Prawns", "Mutton", "Sausages", "Tuna", "Salmon", "Crab"],
    "Snacks":          ["Chips", "Biscuits", "Chocolates", "Nuts", "Crackers", "Wafers", "Candy", "Popcorn", "Pretzels", "Granola Bars"],
    "Household":       ["Detergent", "Soap", "Shampoo", "Toothpaste", "Tissues", "Toilet Paper", "Cleaners", "Fabric Softener", "Dish Soap", "Air Freshener"],
    "Rice & Grains":   ["White Rice", "Red Rice", "Pasta", "Noodles", "Flour", "Oats", "Cereals", "Quinoa", "Barley", "Semolina"],
    "Spices":          ["Chili Powder", "Curry Powder", "Turmeric", "Cinnamon", "Pepper", "Cardamom", "Salt", "Cumin", "Coriander", "Fenugreek"],
    "Frozen Foods":    ["Frozen Vegetables", "Frozen Fries", "Frozen Fish", "Frozen Chicken", "Frozen Pizza", "Frozen Dumplings", "Frozen Corn", "Frozen Peas"],
    "Canned Foods":    ["Canned Fish", "Canned Beans", "Canned Fruits", "Canned Vegetables", "Sauces", "Canned Corn", "Canned Tomatoes", "Canned Soup"],
    "Baby Products":   ["Diapers", "Baby Food", "Baby Powder", "Baby Soap", "Wet Wipes", "Baby Lotion", "Baby Shampoo", "Baby Cereal"],
    "Personal Care":   ["Face Wash", "Moisturizer", "Deodorant", "Perfume", "Hair Oil", "Body Lotion", "Sunscreen", "Lip Balm", "Face Mask", "Toner"],
    "Breakfast Items": ["Jam", "Honey", "Peanut Butter", "Spreads", "Breakfast Cereals", "Maple Syrup", "Granola", "Muesli"],
}

BRANDS = [
    "Keells", "Anchor", "Maliban", "Munchee", "Elephant House",
    "Kotmale", "Coca-Cola", "Nestlé", "Unilever", "Prima",
    "MD", "CBL", "Raigam", "Tiara", "Richlife",
    "Generic", "Store Brand", "Imported"
]

SL_CITIES = [
    "Colombo", "Kandy", "Galle", "Jaffna", "Negombo",
    "Kurunegala", "Anuradhapura", "Trincomalee", "Batticaloa",
    "Matara", "Ratnapura", "Badulla", "Nuwara Eliya", "Kalutara",
    "Gampaha", "Moratuwa", "Kilinochchi", "Vavuniya"
]

CUSTOMER_SEGMENTS = ["frequent_shoppers", "regular_shoppers", "occasional_shoppers", "rare_shoppers"]
SEG_WEIGHTS = [0.20, 0.45, 0.25, 0.10]

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]

FIRST_NAMES = [
    "Allison", "Angie", "Cristian", "Abigail", "Gabrielle", "Monica", "Shannon", "Daniel",
    "Joel", "Andrew", "Jennifer", "Kimberly", "Zachary", "Rebecca", "Tricia", "Patricia",
    "Mark", "Linda", "David", "Kim", "Lisa", "Jessica", "Crystal", "Timothy", "Victoria",
    "Connor", "Angela", "Tammy", "Carmen", "Michael", "Lauren", "Kevin", "Cynthia",
    "Sarah", "Anil", "Priya", "Kavinda", "Dilshan", "Sachini", "Nethmi", "Tharaka",
    "Chamara", "Nimal", "Sunil", "Kumari", "Roshan", "Isuru", "Thilini", "Buddhika"
]
LAST_NAMES = [
    "Hill", "Henderson", "Santos", "Shaffer", "Davis", "Herrera", "Ray", "Adams",
    "Nelson", "Stewart", "Rocha", "Burgess", "Hicks", "Garcia", "Brooks", "Peterson",
    "Perera", "Silva", "Fernando", "Jayasinghe", "Gunasekara", "Rajapaksa", "Wickramasinghe",
    "De Silva", "Seneviratne", "Dissanayake", "Bandara", "Herath", "Wijesinghe", "Amarasinghe"
]

PROMOTION_TYPES = [
    "seasonal_discount", "flash_sale", "loyalty_reward", "new_customer",
    "clearance", "bundle_deal", "category_promotion", "weekend_deal"
]

PRODUCT_SCOPES = ["all_customers", "segment_targeted", "new_customers", "loyal_customers"]

ORDER_STATUSES = ["pending", "confirmed", "processing", "shipped", "delivered", "cancelled"]
ORDER_STATUS_WEIGHTS = [0.05, 0.10, 0.10, 0.10, 0.60, 0.05]


# ─────────────────────────── Generators ───────────────────────

def generate_categories():
    """Returns list of category dicts matching the Category model"""
    rows = []
    for name, meta in CATEGORIES_DEF.items():
        rows.append({
            "id": new_uuid(),
            "name": name,
            "name_si": None,  # optional Sinhala name
            "created_at": fmt_ts(rand_date(datetime(2024, 1, 1), START_DATE)),
            "updated_at": fmt_ts(rand_date(datetime(2024, 1, 1), START_DATE)),
        })
    return rows


def generate_products(categories):
    """Returns list of product dicts matching the Product model"""
    cat_lookup = {c["name"]: c for c in categories}
    rows = []
    sku_counter = 1
    cats = list(CATEGORIES_DEF.keys())
    # distribute products across categories
    prods_per_cat = NUM_PRODUCTS // len(cats)
    remainder = NUM_PRODUCTS - prods_per_cat * len(cats)

    for i, cat_name in enumerate(cats):
        cat = cat_lookup[cat_name]
        count = prods_per_cat + (1 if i < remainder else 0)
        available_names = PRODUCT_NAMES_BY_CAT.get(cat_name, [])
        used = set()
        for _ in range(count):
            base_name = random.choice(available_names) if available_names else f"{cat_name} Item"
            brand = random.choice(BRANDS)
            # Make name unique
            suffix = ""
            attempt = 0
            while f"{brand} {base_name}{suffix}" in used and attempt < 20:
                suffix = f" {attempt + 1}"
                attempt += 1
            prod_name = f"{brand} {base_name}{suffix}"
            used.add(prod_name)

            lo, hi = CATEGORIES_DEF[cat_name]["price_range"]
            price = round(random.uniform(lo, hi), 2)
            stock = random.randint(0, 500)
            pf = CATEGORIES_DEF[cat_name]["purchase_frequency"]

            rows.append({
                "id": new_uuid(),          # maps to productId in Prisma
                "sku": f"SKU{sku_counter:06d}",
                "name": prod_name,
                "name_si": None,
                "description": f"Quality {base_name} by {brand}",
                "description_si": None,
                "category_id": cat["id"],
                "price": str(price),
                "stock_quantity": stock,
                "brand": brand,
                "purchase_frequency": pf,
                "image_url": None,
                "is_active": True,
                "created_by": None,
                "created_at": fmt_ts(rand_date(datetime(2024, 1, 1), START_DATE)),
                "updated_at": fmt_ts(rand_date(datetime(2024, 1, 1), START_DATE)),
                # helper fields (not in DB, used for data gen)
                "_category_name": cat_name,
            })
            sku_counter += 1
    return rows


def generate_users():
    """Returns list of user dicts matching the user model (auth schema)"""
    rows = []
    for i in range(1, NUM_USERS + 1):
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        name = f"{fn} {ln}"
        email = f"user{i:04d}@example.com"
        age = random.randint(18, 75) if random.random() > 0.05 else None
        gender = random.choices(GENDERS, GENDER_WEIGHTS)[0] if random.random() > 0.05 else None
        city = random.choice(SL_CITIES) if random.random() > 0.05 else None
        seg = random.choices(CUSTOMER_SEGMENTS, SEG_WEIGHTS)[0] if random.random() > 0.05 else None
        created = rand_date(datetime(2023, 1, 1), END_DATE)

        rows.append({
            "id": new_uuid(),
            "name": name,
            "email": email,
            "emailVerified": random.random() > 0.1,
            "image": None,
            "age": age,
            "gender": gender,
            "City": city,
            "mobileNumber": f"07{random.randint(10000000, 99999999)}",
            "customerSegment": seg,
            "role": "user",
            "banned": False,
            "banReason": None,
            "banExpires": None,
            "createdAt": fmt_ts(created),
            "updatedAt": fmt_ts(created),
        })
    return rows


def generate_promotions(products):
    """Returns list of promotion dicts matching the Promotion model"""
    rows = []
    active_prods = [p for p in products if p["is_active"]]
    for _ in range(NUM_PROMOTIONS):
        prod = random.choice(active_prods)
        start = rand_date(START_DATE, datetime(2025, 10, 1))
        duration = random.choice([3, 7, 14, 30])
        end = start + timedelta(days=duration)

        discount_pct = round(random.choice([5, 10, 15, 20, 25, 30, 40, 50]), 2)
        is_targeted = random.random() > 0.5
        scope = random.choice(PRODUCT_SCOPES) if is_targeted else "all_customers"
        ptype = random.choice(PROMOTION_TYPES)
        status = "active" if end > END_DATE else ("active" if random.random() > 0.2 else "expired")

        rows.append({
            "promotion_id": new_uuid(),
            "product_id": prod["id"],
            "discount_percentage": str(discount_pct),
            "start_date": fmt_ts(start),
            "end_date": fmt_ts(end),
            "promotion_type": ptype,
            "is_targetted_promotion": is_targeted,
            "product_scope": scope,
            "status": status,
            "created_at": fmt_ts(start - timedelta(days=1)),
            "updated_at": fmt_ts(start),
        })
    return rows


def _get_active_promo(promotions_by_product, product_id, txn_date):
    """Return a matching promotion or None"""
    promos = promotions_by_product.get(product_id, [])
    valid = [p for p in promos if
             datetime.strptime(p["start_date"], "%Y-%m-%d %H:%M:%S+00") <= txn_date <=
             datetime.strptime(p["end_date"], "%Y-%m-%d %H:%M:%S+00")]
    if not valid:
        return None
    return random.choice(valid) if random.random() < 0.25 else None   # 25% chance promo is applied


def generate_orders_and_transactions(users, products, promotions):
    """
    Generate Orders + Transactions (line items).
    
    Matches:
    - Order: id, orderNumber, userId, status, subtotal, discount, tax, total
    - Transaction: transactionId, orderId, invoiceNo, customerId, productId,
                   quantity, unitPrice, totalAmount, transactionDate, discountAmount, promotionId
    """
    # Build lookup: product_id -> list of promos
    promos_by_product = {}
    for p in promotions:
        pid = p["product_id"]
        promos_by_product.setdefault(pid, []).append(p)

    # Build product price lookup
    price_lookup = {p["id"]: float(p["price"]) for p in products}

    # Customer segment → avg transactions / month
    seg_txn_rate = {
        "frequent_shoppers": 12,
        "regular_shoppers": 6,
        "occasional_shoppers": 3,
        "rare_shoppers": 1,
    }

    orders = []
    transactions = []
    invoice_counter = 1
    order_number_counter = 1

    # Spread transactions naturally across users
    # Assign each user a transaction "budget" proportional to their segment
    user_budgets = {}
    total_weight = 0
    for u in users:
        seg = u.get("customerSegment") or "occasional_shoppers"
        w = seg_txn_rate.get(seg, 3)
        user_budgets[u["id"]] = w
        total_weight += w

    # Normalise budgets to sum to NUM_TRANSACTIONS
    for uid in user_budgets:
        user_budgets[uid] = max(1, int(round(user_budgets[uid] / total_weight * NUM_TRANSACTIONS)))

    # Build user order per session
    for user in users:
        uid = user["id"]
        n_transactions = user_budgets.get(uid, 3)

        # Group transactions into shopping sessions (1–10 items each)
        remaining = n_transactions
        while remaining > 0:
            session_size = min(remaining, random.randint(1, 14))
            remaining -= session_size

            txn_date = rand_date()

            # Create order
            order_id = new_uuid()
            order_number = f"ORD{order_number_counter:08d}"
            order_number_counter += 1
            order_status = random.choices(ORDER_STATUSES, ORDER_STATUS_WEIGHTS)[0]

            # Pick random products for this session
            session_products = random.choices(products, k=session_size)
            subtotal = 0.0
            order_discount = 0.0
            order_transactions = []

            for prod in session_products:
                pid = prod["id"]
                unit_price = price_lookup.get(pid, 100.0)
                qty = random.randint(1, 5)
                promo = _get_active_promo(promos_by_product, pid, txn_date)

                discount_amount = 0.0
                promo_id = None
                if promo:
                    disc_pct = float(promo["discount_percentage"])
                    discount_amount = round(unit_price * qty * disc_pct / 100, 2)
                    promo_id = promo["promotion_id"]

                total_amount = round(unit_price * qty - discount_amount, 2)
                subtotal += unit_price * qty
                order_discount += discount_amount

                order_transactions.append({
                    "transaction_id": new_uuid(),
                    "order_id": order_id,
                    "invoice_no": f"INV{invoice_counter:08d}",
                    "customer_id": uid,
                    "product_id": pid,
                    "quantity": qty,
                    "unit_price": str(round(unit_price, 2)),
                    "total_amount": str(total_amount),
                    "transaction_date": fmt_ts(txn_date),
                    "discount_amount": str(discount_amount),
                    "promotion_id": promo_id,
                })
                invoice_counter += 1

            tax = round(subtotal * 0.05, 2)   # 5% VAT
            total = round(subtotal - order_discount + tax, 2)

            orders.append({
                "id": order_id,
                "order_number": order_number,
                "user_id": uid,
                "status": order_status,
                "subtotal": str(round(subtotal, 2)),
                "discount": str(round(order_discount, 2)),
                "tax": str(tax),
                "total": str(total),
                "shipping_address": f"{user.get('City', 'Colombo')}, Sri Lanka",
                "billing_address": f"{user.get('City', 'Colombo')}, Sri Lanka",
                "notes": None,
                "created_at": fmt_ts(txn_date),
                "updated_at": fmt_ts(txn_date),
            })
            transactions.extend(order_transactions)

    return orders, transactions


# ─────────────────────────── CSV Writers ─────────────────────

def write_csv(filepath, rows, fieldnames=None):
    if not rows:
        print(f"  Warning: No rows for {filepath}")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):,} rows -> {os.path.basename(filepath)}")


# ─────────────────────────── Main ────────────────────────────

def main():
    print("=" * 70)
    print(" DB-ALIGNED DATASET GENERATOR")
    print(" Matches Prisma schema: e_commerce_platform_schema.prizma")
    print("=" * 70)

    print("\n1. Generating categories...")
    categories = generate_categories()
    write_csv(
        os.path.join(OUTPUT_DIR, "categories.csv"),
        categories,
        fieldnames=["id", "name", "name_si", "created_at", "updated_at"]
    )

    print("\n2. Generating products...")
    products = generate_products(categories)
    write_csv(
        os.path.join(OUTPUT_DIR, "products.csv"),
        products,
        fieldnames=["id", "sku", "name", "name_si", "description", "description_si",
                    "category_id", "price", "stock_quantity", "brand",
                    "purchase_frequency", "image_url", "is_active",
                    "created_by", "created_at", "updated_at"]
    )

    print("\n3. Generating users...")
    users = generate_users()
    write_csv(
        os.path.join(OUTPUT_DIR, "users.csv"),
        users,
        fieldnames=["id", "name", "email", "emailVerified", "image", "age", "gender",
                    "City", "mobileNumber", "customerSegment", "role",
                    "banned", "banReason", "banExpires", "createdAt", "updatedAt"]
    )

    print("\n4. Generating promotions...")
    promotions = generate_promotions(products)
    write_csv(
        os.path.join(OUTPUT_DIR, "promotions.csv"),
        promotions,
        fieldnames=["promotion_id", "product_id", "discount_percentage", "start_date",
                    "end_date", "promotion_type", "is_targetted_promotion",
                    "product_scope", "status", "created_at", "updated_at"]
    )

    print("\n5. Generating orders and transactions...")
    orders, transactions = generate_orders_and_transactions(users, products, promotions)
    write_csv(
        os.path.join(OUTPUT_DIR, "orders.csv"),
        orders,
        fieldnames=["id", "order_number", "user_id", "status", "subtotal",
                    "discount", "tax", "total", "shipping_address",
                    "billing_address", "notes", "created_at", "updated_at"]
    )
    write_csv(
        os.path.join(OUTPUT_DIR, "transactions.csv"),
        transactions,
        fieldnames=["transaction_id", "order_id", "invoice_no", "customer_id",
                    "product_id", "quantity", "unit_price", "total_amount",
                    "transaction_date", "discount_amount", "promotion_id"]
    )

    print(f"\n{'=' * 70}")
    print(" GENERATION COMPLETE")
    print(f" Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    print(f"\n Summary:")
    print(f"  Categories  : {len(categories):>6,}")
    print(f"  Products    : {len(products):>6,}")
    print(f"  Users       : {len(users):>6,}")
    print(f"  Promotions  : {len(promotions):>6,}")
    print(f"  Orders      : {len(orders):>6,}")
    print(f"  Transactions: {len(transactions):>6,}")


if __name__ == "__main__":
    main()
