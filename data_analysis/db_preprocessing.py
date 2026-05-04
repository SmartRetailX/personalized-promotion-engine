"""
DB-Aligned Data Preprocessor
Reads CSVs from data/raw_db/ (schema-matched data) and creates ML features.

Key differences from old preprocessor:
- customer_id = UUID string (from user.id)
- product_id  = UUID string (from Product.id)
- promotion_id = UUID string (from Promotion.promotionId)
- Field names match Prisma snake_case column names
- category_id is a UUID, category name resolved via join
- No StoreID (not in schema)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
import os


class DBAlignedPreprocessor:
    """
    Reads DB-aligned CSVs and prepares features for the ML pipeline.
    Drop-in replacement for DataPreprocessor in the promotion engine.
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(script_dir), "data", "raw_db")
        self.data_dir = data_dir

        # Public DataFrames (mirror old API)
        self.customers = None          # maps to 'users' table
        self.products = None
        self.transactions = None
        self.promotions = None
        self.categories = None

        self.category_encoder = LabelEncoder()

    # ─────────────── load ────────────────────────────────────

    def load_data(self):
        """Load all DB-aligned CSVs."""
        print("Loading DB-aligned datasets...")

        # ── users (auth.user)
        users_path = os.path.join(self.data_dir, "users.csv")
        raw_users = pd.read_csv(users_path)
        # Normalise to internal names used by the ML pipeline
        self.customers = raw_users.rename(columns={
            "id":              "CustomerID",
            "name":            "Name",
            "email":           "Email",
            "age":             "Age",
            "gender":          "Gender",
            "City":            "Location",
            "customerSegment": "CustomerSegment",
            "createdAt":       "RegistrationDate",
        })
        self.customers["Age"] = pd.to_numeric(self.customers["Age"], errors="coerce").fillna(30).astype(int)
        self.customers["RegistrationDate"] = pd.to_datetime(
            self.customers["RegistrationDate"], errors="coerce", utc=True
        )

        # ── categories (core.categories)
        cats_path = os.path.join(self.data_dir, "categories.csv")
        self.categories = pd.read_csv(cats_path)

        # ── products (core.products)
        prods_path = os.path.join(self.data_dir, "products.csv")
        raw_prods = pd.read_csv(prods_path)
        # Join category name
        raw_prods = raw_prods.merge(
            self.categories[["id", "name"]].rename(columns={"id": "category_id", "name": "Category"}),
            on="category_id", how="left"
        )
        self.products = raw_prods.rename(columns={
            "id":                "ProductID",
            "name":              "ProductName",
            "price":             "Price",
            "stock_quantity":    "StockQuantity",
            "brand":             "Brand",
            "purchase_frequency": "PurchaseFrequency",
        })
        self.products["Price"] = pd.to_numeric(self.products["Price"], errors="coerce").fillna(0)

        # ── promotions (core.promotions)
        promos_path = os.path.join(self.data_dir, "promotions.csv")
        raw_promos = pd.read_csv(promos_path)
        self.promotions = raw_promos.rename(columns={
            "promotion_id":          "PromotionID",
            "product_id":            "ProductID",
            "discount_percentage":   "DiscountPercentage",
            "start_date":            "StartDate",
            "end_date":              "EndDate",
            "promotion_type":        "PromotionType",
            "is_targetted_promotion": "IsTargetted",
            "product_scope":         "ProductScope",
        })

        # ── transactions (core.transactions)
        txns_path = os.path.join(self.data_dir, "transactions.csv")
        raw_txns = pd.read_csv(txns_path)
        self.transactions = raw_txns.rename(columns={
            "transaction_id":  "TransactionID",
            "order_id":        "OrderID",
            "invoice_no":      "InvoiceNo",
            "customer_id":     "CustomerID",
            "product_id":      "ProductID",
            "quantity":        "Quantity",
            "unit_price":      "UnitPrice",
            "total_amount":    "TotalAmount",
            "transaction_date": "TransactionDate",
            "discount_amount": "DiscountedAmount",
            "promotion_id":    "PromotionID",
        })
        self.transactions["TransactionDate"] = pd.to_datetime(
            self.transactions["TransactionDate"], errors="coerce", utc=True
        ).dt.tz_localize(None)   # strip tz for arithmetic compatibility
        self.transactions["TotalAmount"] = pd.to_numeric(self.transactions["TotalAmount"], errors="coerce").fillna(0)
        self.transactions["DiscountedAmount"] = pd.to_numeric(self.transactions["DiscountedAmount"], errors="coerce").fillna(0)
        self.transactions["Quantity"] = pd.to_numeric(self.transactions["Quantity"], errors="coerce").fillna(1).astype(int)
        # Fill NULL promotionID with "None" string for compatibility
        self.transactions["PromotionID"] = self.transactions["PromotionID"].fillna("None")

        # ── fit category encoder
        self.category_encoder.fit(self.products["Category"].dropna().unique())

        print(f"  Users       : {len(self.customers):,}")
        print(f"  Products    : {len(self.products):,}")
        print(f"  Promotions  : {len(self.promotions):,}")
        print(f"  Transactions: {len(self.transactions):,}")
        print(f"  Categories  : {list(self.category_encoder.classes_)}")

    # ─────────────── feature helpers ────────────────────────

    def calculate_customer_features_for_window(self, transactions, reference_date):
        """
        Mirror of old DataPreprocessor API.
        Returns DataFrame with columns:
          CustomerID, purchase_frequency, total_spent, avg_transaction,
          recency_days, promo_purchases, promo_response_rate
        """
        if len(transactions) == 0:
            return pd.DataFrame(columns=["CustomerID", "purchase_frequency",
                                         "total_spent", "avg_transaction",
                                         "recency_days", "promo_response_rate"])

        cust_feats = transactions.groupby("CustomerID").agg(
            purchase_frequency=("TransactionID", "count"),
            total_spent=("TotalAmount", "sum"),
            avg_transaction=("TotalAmount", "mean"),
            recency_days=("TransactionDate", lambda x: (reference_date - x.max()).days),
        ).reset_index()

        # Promo response
        promo_trans = transactions[transactions["PromotionID"] != "None"]
        promo_response = promo_trans.groupby("CustomerID").size().reset_index(name="promo_purchases")
        cust_feats = cust_feats.merge(promo_response, on="CustomerID", how="left")
        cust_feats["promo_purchases"] = cust_feats["promo_purchases"].fillna(0)
        cust_feats["promo_response_rate"] = (
            cust_feats["promo_purchases"] / cust_feats["purchase_frequency"]
        )
        return cust_feats

    def create_customer_features(self, transactions=None, reference_date=None):
        """
        Create customer-level features aligned with the original pipeline.

        Returns columns used by PromotionOptimizer:
          total_spent, avg_transaction_value, avg_discount_per_transaction,
          promo_response_rate, CustomerSegment
        """
        print("\nCreating customer features (DB-aligned)...")

        if transactions is None:
            transactions = self.transactions
        if reference_date is None:
            reference_date = transactions["TransactionDate"].max()

        if len(transactions) == 0:
            return pd.DataFrame(columns=[
                "CustomerID", "purchase_frequency", "total_spent",
                "avg_transaction_value", "std_transaction_value",
                "recency_days", "total_items_purchased",
                "avg_items_per_transaction", "total_discounts_received",
                "avg_discount_per_transaction", "promo_purchases",
                "promo_response_rate", "Name", "Age", "Gender",
                "Location", "CustomerSegment",
            ])

        customer_features = transactions.groupby("CustomerID").agg(
            purchase_frequency=("TransactionID", "count"),
            total_spent=("TotalAmount", "sum"),
            avg_transaction_value=("TotalAmount", "mean"),
            std_transaction_value=("TotalAmount", "std"),
            recency_days=("TransactionDate", lambda x: (reference_date - x.max()).days),
            total_items_purchased=("Quantity", "sum"),
            avg_items_per_transaction=("Quantity", "mean"),
            total_discounts_received=("DiscountedAmount", "sum"),
            avg_discount_per_transaction=("DiscountedAmount", "mean"),
        ).reset_index()

        customer_features["std_transaction_value"] = customer_features[
            "std_transaction_value"
        ].fillna(0)

        promo_trans = transactions[transactions["PromotionID"] != "None"]
        promo_response = promo_trans.groupby("CustomerID").size().reset_index(
            name="promo_purchases"
        )
        customer_features = customer_features.merge(
            promo_response, on="CustomerID", how="left"
        )
        customer_features["promo_purchases"] = customer_features[
            "promo_purchases"
        ].fillna(0)
        customer_features["promo_response_rate"] = (
            customer_features["promo_purchases"] / customer_features["purchase_frequency"]
        )

        customer_features = customer_features.merge(
            self.customers[["CustomerID", "Name", "Age", "Gender", "Location", "CustomerSegment"]],
            on="CustomerID",
            how="left",
        )

        return customer_features

    def calculate_category_affinity(self, transactions):
        """
        Mirror of old DataPreprocessor API.
        Returns dict: {CustomerID: {CategoryName: affinity_score}}
        """
        if len(transactions) == 0:
            return {}

        trans_with_cat = transactions.merge(
            self.products[["ProductID", "Category"]], on="ProductID", how="left"
        )
        cat_spending = (
            trans_with_cat.groupby(["CustomerID", "Category"])["TotalAmount"]
            .sum()
            .reset_index()
        )
        total_spending = cat_spending.groupby("CustomerID")["TotalAmount"].transform("sum")
        cat_spending["affinity"] = cat_spending["TotalAmount"] / (total_spending + 1e-9)

        affinity_dict = {}
        for _, row in cat_spending.iterrows():
            cid = row["CustomerID"]
            if cid not in affinity_dict:
                affinity_dict[cid] = {}
            affinity_dict[cid][row["Category"]] = row["affinity"]
        return affinity_dict

    def create_customer_product_matrix(self, end_date=None):
        """
        Creates interaction matrix for CF model.
        FIX #5: end_date prevents future leakage.
        FIX #6: log-scaled interaction scores.
        """
        print("Creating customer-product interaction matrix ...")
        txns = self.transactions
        if end_date is not None:
            txns = txns[txns["TransactionDate"] < end_date]
            print(f"  Using transactions up to {end_date.date() if hasattr(end_date, 'date') else end_date}")

        interactions = txns.groupby(["CustomerID", "ProductID"]).agg(
            total_quantity=("Quantity", "sum"),
            total_spent=("TotalAmount", "sum"),
            purchase_count=("TransactionID", "count"),
        ).reset_index()
        interactions["interaction_score"] = np.log1p(interactions["purchase_count"])
        print(f"  {len(interactions):,} interaction records created")
        return interactions

    def create_time_aware_training_data(self, observation_days=90, prediction_days=14):
        """
        Build labelled training dataset (customer × product -> will_buy_next_N_days?).
        Mirrors old DataPreprocessor API exactly.
        """
        print("\n" + "=" * 70)
        print(" CREATING TIME-AWARE TRAINING DATA (DB-ALIGNED)")
        print("=" * 70)

        min_date = self.transactions["TransactionDate"].min()
        max_date = self.transactions["TransactionDate"].max()
        print(f"\nDate range: {min_date.date()} to {max_date.date()}")
        print(f"Observation window: {observation_days} days | Prediction window: {prediction_days} days")

        total_days = (max_date - min_date).days
        if total_days < observation_days + prediction_days:
            print("WARNING: Not enough data for proper time split - adjusting windows")
            observation_days = total_days // 3
            prediction_days = total_days // 6

        # Sample products for efficiency (max 30)
        all_products = self.products["ProductID"].unique()
        sampled_products = (
            np.random.choice(all_products, 30, replace=False)
            if len(all_products) > 30 else all_products
        )

        # Build observation points every 30 days
        obs_points = []
        cur = min_date + timedelta(days=observation_days)
        while cur + timedelta(days=prediction_days) <= max_date:
            obs_points.append(cur)
            cur += timedelta(days=30)
        print(f"Observation points: {len(obs_points)}")

        samples = []
        for oi, obs_date in enumerate(obs_points):
            print(f"  Point {oi+1}/{len(obs_points)}: {obs_date.date()}")

            obs_start = obs_date - timedelta(days=observation_days)
            pred_end = obs_date + timedelta(days=prediction_days)

            obs_txns = self.transactions[
                (self.transactions["TransactionDate"] >= obs_start) &
                (self.transactions["TransactionDate"] < obs_date)
            ]
            pred_txns = self.transactions[
                (self.transactions["TransactionDate"] >= obs_date) &
                (self.transactions["TransactionDate"] < pred_end)
            ]

            if len(obs_txns) == 0:
                continue

            cust_feats = self.calculate_customer_features_for_window(obs_txns, obs_date)
            cat_affinity = self.calculate_category_affinity(obs_txns)

            # Set of (customer, product) in prediction window for fast lookup
            future_buys = set(
                zip(pred_txns["CustomerID"], pred_txns["ProductID"])
            )

            for _, cf_row in cust_feats.iterrows():
                cid = cf_row["CustomerID"]
                cust_age = self.customers[self.customers["CustomerID"] == cid]["Age"].values
                cust_age = int(cust_age[0]) if len(cust_age) > 0 else 30
                cust_cats = cat_affinity.get(cid, {})

                for pid in sampled_products:
                    prod_row = self.products[self.products["ProductID"] == pid]
                    if len(prod_row) == 0:
                        continue
                    prod_row = prod_row.iloc[0]
                    prod_cat = prod_row["Category"]

                    if prod_cat in self.category_encoder.classes_:
                        cat_encoded = int(self.category_encoder.transform([prod_cat])[0])
                    else:
                        cat_encoded = -1

                    cat_purchase_count = len(obs_txns[
                        (obs_txns["CustomerID"] == cid) &
                        (obs_txns["ProductID"].isin(
                            self.products[self.products["Category"] == prod_cat]["ProductID"]
                        ))
                    ])

                    target = 1 if (cid, pid) in future_buys else 0

                    samples.append({
                        "CustomerID":                   cid,
                        "ProductID":                    pid,
                        "observation_date":             obs_date,
                        "customer_purchase_frequency":  cf_row["purchase_frequency"],
                        "customer_avg_transaction":     cf_row["avg_transaction"],
                        "customer_recency":             cf_row["recency_days"],
                        "customer_promo_response_rate": cf_row["promo_response_rate"],
                        "customer_age":                 cust_age,
                        "product_price":                float(prod_row["Price"]),
                        "product_category_encoded":     cat_encoded,
                        "category_affinity":            cust_cats.get(prod_cat, 0.0),
                        "category_purchase_count":      cat_purchase_count,
                        "target":                       target,
                    })

        df = pd.DataFrame(samples)
        if len(df) > 0:
            print(f"\nTotal samples: {len(df):,}")
            print(f"Positive: {df['target'].sum():,}  ({df['target'].mean()*100:.2f}%)")
        return df

    def get_train_val_test_split(self, df, val_ratio=0.15, test_ratio=0.15):
        """Time-ordered train / val / test split. Mirrors old API."""
        print("\nCreating time-based train/val/test split ...")
        df = df.sort_values("observation_date")
        n = len(df)
        train_end = int(n * (1 - val_ratio - test_ratio))
        val_end = int(n * (1 - test_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print(f"  Train : {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
        print(f"  Val   : {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
        print(f"  Test  : {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
        return train_df, val_df, test_df


# ──────────────────────── quick test ────────────────────────

def main():
    prep = DBAlignedPreprocessor()
    prep.load_data()
    interactions = prep.create_customer_product_matrix()
    print(f"\nSample interaction row:\n{interactions.head(2)}")


if __name__ == "__main__":
    main()
