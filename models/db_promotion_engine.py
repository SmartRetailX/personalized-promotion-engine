"""
DB-Aligned Personalized Promotion Engine
=========================================
End-to-end promotion targeting system that works with the Prisma schema.

Key differences from the original promotion_engine.py:
- Uses DBAlignedPreprocessor instead of DataPreprocessor
- Loads db_purchase_prediction_model.pkl / db_collaborative_filtering_model.pkl
- All IDs are UUIDs (product_id, customer_id = user.id)
- Output fields match what the Next.js API expects

Pipeline (same CF+ML logic):
  Step 1: CF finds candidate users (similar to product buyers)
  Step 2: ML scores each candidate's purchase probability
  Step 3: Hybrid ranking, threshold filtering, top-N returned

API usage (can be called from FastAPI / Next.js backend):
  engine = DBAlignedPromotionEngine()
  engine.load_models()
  
  # Get targets for a product UUID
  targets = engine.generate_promotion_targets(product_uuid, top_n=50)
  
  # Returns DataFrame with columns:
  #   customer_id, product_id, purchase_probability, cf_score,
  #   hybrid_score, above_threshold, category_affinity, targeting_method
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.purchase_prediction import PurchasePredictionModel
from models.collaborative_filtering import CollaborativeFilteringModel
from models.promotion_optimizer import PromotionOptimizer
from data_analysis.db_preprocessing import DBAlignedPreprocessor


class DBAlignedPromotionEngine:
    """
    End-to-end promotion targeting system aligned to the Prisma DB schema.
    """

    # Model file names
    PURCHASE_MODEL_FILE = "db_purchase_prediction_model.pkl"
    CF_MODEL_FILE = "db_collaborative_filtering_model.pkl"

    def __init__(self, models_dir="models"):
        if not os.path.isabs(models_dir):
            models_dir = os.path.join(project_root, models_dir)
        self.models_dir = models_dir
        self.purchase_model: PurchasePredictionModel | None = None
        self.cf_model: CollaborativeFilteringModel | None = None
        self.optimizer: PromotionOptimizer | None = None
        self.preprocessor: DBAlignedPreprocessor | None = None

    # ─────────────────────────── setup ──────────────────────

    def load_models(self):
        """Load ML + CF models and preprocessor."""
        print("Loading DB-aligned models ...")

        # Purchase prediction
        pp_path = os.path.join(self.models_dir, self.PURCHASE_MODEL_FILE)
        if os.path.exists(pp_path):
            self.purchase_model = PurchasePredictionModel.load_model(pp_path)
        else:
            print(f"  WARNING: {self.PURCHASE_MODEL_FILE} not found. Run train_db_aligned.py first.")

        # Collaborative filtering
        cf_path = os.path.join(self.models_dir, self.CF_MODEL_FILE)
        if os.path.exists(cf_path):
            self.cf_model = CollaborativeFilteringModel.load_model(cf_path)
        else:
            print(f"  WARNING: {self.CF_MODEL_FILE} not found. Run train_db_aligned.py first.")

        # Preprocessor (reads raw_db CSVs)
        self.preprocessor = DBAlignedPreprocessor()
        self.preprocessor.load_data()

        # Optimizer (optional - for discount optimisation)
        self.optimizer = PromotionOptimizer()
        txn_path = os.path.join(project_root, "data", "raw_db", "transactions.csv")
        processed_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        cust_path = os.path.join(processed_dir, "customer_features.csv")
        if os.path.exists(txn_path):
            # Build full customer_features compatible with PromotionOptimizer
            cust_feats = self.preprocessor.create_customer_features(
                self.preprocessor.transactions,
                self.preprocessor.transactions["TransactionDate"].max()
            )
            cust_feats.to_csv(cust_path, index=False)
            self.optimizer.load_data(cust_path, txn_path)

        print("* Models loaded.\n")

    # ─────────────────────────── main API ───────────────────

    def generate_promotion_targets(self, product_id: str, top_n: int = 50) -> pd.DataFrame:
        """
        CF + ML pipeline to find best customers for a product promotion.

        Args:
            product_id : UUID string (Product.id in DB)
            top_n      : Maximum customers to return

        Returns:
            DataFrame with columns:
              customer_id, product_id, purchase_probability, cf_score,
              hybrid_score, above_threshold, category_affinity, targeting_method
        """
        print("\n" + "=" * 70)
        print(" GENERATING PROMOTION TARGETS (CF + ML PIPELINE)")
        print("=" * 70)

        if self.purchase_model is None or self.cf_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # ── Product info ──────────────────────────────────
        product = self.preprocessor.products[
            self.preprocessor.products["ProductID"] == product_id
        ]
        if len(product) == 0:
            print(f"ERROR: product_id {product_id} not found.")
            return pd.DataFrame()

        product = product.iloc[0]
        print(f"\nProduct : {product['ProductName']}")
        print(f"Category: {product['Category']}")
        print(f"Price   : {float(product['Price']):,.2f}")

        # ── STEP 1: CF candidates ─────────────────────────
        print("\n[STEP 1] CF - finding candidate customers ...")
        cf_results = self.cf_model.find_customers_for_product(
            product_id, n=top_n * 3, return_scores=True
        )
        print(f"  Found {len(cf_results)} CF candidates")

        if not cf_results:
            print("  WARNING: No CF candidates - falling back to all users")
            cf_candidates = self.preprocessor.customers["CustomerID"].tolist()[:top_n * 3]
            cf_scores = {cid: 0.5 for cid in cf_candidates}
        else:
            cf_candidates = [r["CustomerID"] for r in cf_results]
            cf_scores = {r["CustomerID"]: r["cf_score"] for r in cf_results}

        # ── STEP 2: ML feature preparation ───────────────
        print("\n[STEP 2] Preparing ML features ...")

        max_date = self.preprocessor.transactions["TransactionDate"].max()
        obs_start = max_date - pd.Timedelta(days=180)

        obs_txns = self.preprocessor.transactions[
            self.preprocessor.transactions["TransactionDate"] >= obs_start
        ]

        cust_feats_obs = self.preprocessor.calculate_customer_features_for_window(
            obs_txns, max_date
        )
        cat_affinity = self.preprocessor.calculate_category_affinity(obs_txns)

        prod_cat = product["Category"]
        if prod_cat in self.preprocessor.category_encoder.classes_:
            cat_encoded = int(self.preprocessor.category_encoder.transform([prod_cat])[0])
        else:
            cat_encoded = -1
            print(f"  WARNING: unknown category '{prod_cat}', using -1")

        # Fallback values for cold customers (use population medians)
        if len(cust_feats_obs) > 0:
            fallback = {
                "purchase_frequency": cust_feats_obs["purchase_frequency"].median(),
                "avg_transaction":    cust_feats_obs["avg_transaction"].median(),
                "recency_days":       cust_feats_obs["recency_days"].median(),
                "promo_response_rate": cust_feats_obs["promo_response_rate"].median(),
            }
        else:
            fallback = {"purchase_frequency": 5, "avg_transaction": 500,
                        "recency_days": 30, "promo_response_rate": 0.1}

        cat_product_ids = set(
            self.preprocessor.products[
                self.preprocessor.products["Category"] == prod_cat
            ]["ProductID"]
        )

        candidate_features = []
        cold_count = 0
        for cid in cf_candidates:
            cf_row = cust_feats_obs[cust_feats_obs["CustomerID"] == cid]
            if len(cf_row) == 0:
                cold_count += 1
                feats = fallback
            else:
                r = cf_row.iloc[0]
                feats = {
                    "purchase_frequency":  r["purchase_frequency"],
                    "avg_transaction":     r["avg_transaction"],
                    "recency_days":        r["recency_days"],
                    "promo_response_rate": r["promo_response_rate"],
                }

            cust_cat_affinity = cat_affinity.get(cid, {})
            age = self.preprocessor.customers[
                self.preprocessor.customers["CustomerID"] == cid
            ]["Age"].values
            age = int(age[0]) if len(age) > 0 else 30

            cat_purchase_count = len(obs_txns[
                (obs_txns["CustomerID"] == cid) &
                (obs_txns["ProductID"].isin(cat_product_ids))
            ])

            candidate_features.append({
                "CustomerID":                    cid,
                "ProductID":                     product_id,
                "customer_purchase_frequency":   feats["purchase_frequency"],
                "customer_avg_transaction":      feats["avg_transaction"],
                "customer_recency":              feats["recency_days"],
                "customer_promo_response_rate":  feats["promo_response_rate"],
                "customer_age":                  age,
                "product_price":                 float(product["Price"]),
                "product_category_encoded":      cat_encoded,
                "category_affinity":             cust_cat_affinity.get(prod_cat, 0.0),
                "category_purchase_count":       cat_purchase_count,
                "cf_score":                      cf_scores.get(cid, 0.0),
            })

        if not candidate_features:
            print("  ERROR: no candidate features built.")
            return pd.DataFrame()

        if cold_count:
            print(f"  Note: {cold_count} cold-start customers included with median fallback features")

        candidate_df = pd.DataFrame(candidate_features)
        print(f"  Feature matrix: {len(candidate_df)} rows")

        # ── STEP 3: ML scoring ────────────────────────────
        print("\n[STEP 3] ML scoring ...")
        X, _, _ = self.purchase_model.prepare_features(candidate_df)

        # Feature alignment check
        if self.purchase_model.feature_cols is not None:
            expected = set(self.purchase_model.feature_cols)
            actual = set(X.columns)
            if expected != actual:
                miss = expected - actual
                extra = actual - expected
                msg = "Feature mismatch!\n"
                if miss:
                    msg += f"  Missing: {miss}\n"
                if extra:
                    msg += f"  Extra  : {extra}\n"
                raise ValueError(msg)

        candidate_df["purchase_probability"] = self.purchase_model.predict_proba(X)

        ML_W, CF_W = 0.70, 0.30
        candidate_df["hybrid_score"] = (
            ML_W * candidate_df["purchase_probability"] +
            CF_W * candidate_df["cf_score"]
        )
        print(f"  Hybrid weights: {ML_W:.0%} ML + {CF_W:.0%} CF")
        print(f"  ML prob range : {candidate_df['purchase_probability'].min():.4f}"
              f" - {candidate_df['purchase_probability'].max():.4f}")

        # ── STEP 4: Threshold & ranking ──────────────────
        print("\n[STEP 4] Threshold filtering and ranking ...")
        percentile_threshold = candidate_df["purchase_probability"].quantile(0.5)
        promotion_threshold = max(0.02, percentile_threshold)
        print(f"  Promotion threshold: {promotion_threshold:.4f}")

        eligible = candidate_df[
            candidate_df["purchase_probability"] >= promotion_threshold
        ].sort_values("hybrid_score", ascending=False)

        top_targets = eligible.head(top_n).copy()
        top_targets["above_threshold"] = True
        top_targets["targeting_method"] = "cf_ml_hybrid"

        print(f"\n  Scored        : {len(candidate_df)}")
        print(f"  Eligible      : {len(eligible)}")
        print(f"  Returned top-N: {len(top_targets)}")

        if len(top_targets) > 0:
            print("\n  Top 5 targets:")
            for _, row in top_targets.head(5).iterrows():
                print(f"    {row['CustomerID'][:8]}..."
                      f"  ML={row['purchase_probability']:.2%}"
                      f"  CF={row['cf_score']:.2f}"
                      f"  Hybrid={row['hybrid_score']:.2%}")

        # Rename CustomerID -> customer_id for consistency with Prisma field names
        top_targets = top_targets.rename(columns={
            "CustomerID": "customer_id",
            "ProductID":  "product_id",
        })

        return top_targets[[
            "customer_id", "product_id",
            "purchase_probability", "cf_score", "hybrid_score",
            "above_threshold", "category_affinity", "targeting_method"
        ]]

    # ─────────────────────────── campaign API ───────────────

    def create_promotion_campaign(
        self,
        product_id: str,
        max_targets: int = 100,
        optimize: bool = True,
    ) -> tuple[pd.DataFrame | None, dict | None]:
        """
        Create a promotion campaign for a product.
        Returns (campaign_df, summary_dict).
        """
        print("\n" + "=" * 70)
        print(" CREATING PROMOTION CAMPAIGN")
        print("=" * 70)

        product = self.preprocessor.products[
            self.preprocessor.products["ProductID"] == product_id
        ]
        if len(product) == 0:
            print(f"ERROR: product {product_id} not found")
            return None, None

        product = product.iloc[0]
        prod_name = product["ProductName"]
        prod_price = float(product["Price"])
        print(f"\nProduct: {prod_name}  |  Price: {prod_price:,.2f}")

        # Get targets (returns customer_id column)
        target_df = self.generate_promotion_targets(product_id, top_n=max_targets * 2)

        if len(target_df) == 0:
            return None, {"product_id": product_id, "product_name": prod_name, "num_customers_targeted": 0}

        # Rename back for optimizer compatibility
        target_df_compat = target_df.rename(columns={"customer_id": "CustomerID", "product_id": "ProductID"})

        if optimize and self.optimizer:
            campaign, summary = self.optimizer.generate_promotion_campaign(
                product_id, prod_name, prod_price, target_df_compat, max_targets
            )
        else:
            campaign = target_df_compat.head(max_targets).copy()
            summary = {
                "product_id": product_id,
                "product_name": prod_name,
                "num_customers_targeted": len(campaign),
            }

        print("\n" + "=" * 70)
        print(" CAMPAIGN CREATED")
        print("=" * 70)
        return campaign, summary

    def compare_personalized_vs_broadcast(
        self,
        product_id: str,
        broadcast_discount: float = 15.0,
    ) -> dict:
        """Compare personalised vs broadcast ROI."""
        print("\n" + "=" * 70)
        print(" PERSONALIZED vs BROADCAST COMPARISON")
        print("=" * 70)

        product = self.preprocessor.products[
            self.preprocessor.products["ProductID"] == product_id
        ].iloc[0]
        price = float(product["Price"])

        # Estimate historical conversion rate from actual data
        promo_txns = self.preprocessor.transactions[
            self.preprocessor.transactions["PromotionID"] != "None"
        ]
        total_users = self.preprocessor.customers["CustomerID"].nunique()
        if len(promo_txns) > 0:
            conv_rate = promo_txns["CustomerID"].nunique() / total_users
        else:
            conv_rate = 0.10

        print(f"\nHistorical conversion rate: {conv_rate:.1%}")

        # Personalised
        pers_campaign, pers_summary = self.create_promotion_campaign(
            product_id, max_targets=100, optimize=True
        )

        # Broadcast
        broadcast_cost = total_users * price * (broadcast_discount / 100) * conv_rate
        broadcast_revenue = total_users * price * conv_rate
        broadcast_profit = broadcast_revenue - broadcast_cost

        print(f"\nBroadcast - {total_users:,} users:")
        print(f"  Cost   : {broadcast_cost:,.2f}")
        print(f"  Revenue: {broadcast_revenue:,.2f}")
        print(f"  Profit : {broadcast_profit:,.2f}")

        if pers_summary and "total_expected_profit" in pers_summary:
            efficiency = pers_summary.get("num_customers_targeted", 0) / total_users * 100
            improvement = (
                (pers_summary["total_expected_profit"] - broadcast_profit)
                / (abs(broadcast_profit) + 1) * 100
            )
            print(f"\nPersonalised reaches {efficiency:.1f}% of users")
            print(f"Profit improvement   : {improvement:.1f}%")

        return {
            "personalized": pers_summary,
            "broadcast": {
                "cost": broadcast_cost,
                "revenue": broadcast_revenue,
                "profit": broadcast_profit,
                "num_customers": total_users,
            },
        }


# ─────────────────────────── Demo ───────────────────────────

def main():
    print("=" * 70)
    print(" DB-ALIGNED PROMOTION ENGINE - DEMO")
    print("=" * 70)

    engine = DBAlignedPromotionEngine()
    engine.load_models()

    if engine.purchase_model is None or engine.cf_model is None:
        print("\nModels are missing. Run train_db_aligned.py from the project root.")
        return

    # Pick a sample product from the DB-aligned data
    sample_product = engine.preprocessor.products[
        engine.preprocessor.products["Category"] == "Bakery"
    ]
    if len(sample_product) == 0:
        sample_product = engine.preprocessor.products
    product_id = sample_product.iloc[0]["ProductID"]

    print(f"\nDemo product: {sample_product.iloc[0]['ProductName']}")
    print(f"UUID        : {product_id}")

    # Generate targets
    targets = engine.generate_promotion_targets(product_id, top_n=50)

    if len(targets) > 0:
        print(f"\n{'─'*50}")
        print("TARGET SUMMARY")
        print(f"{'─'*50}")
        print(f"  Total returned  : {len(targets)}")
        print(f"  Avg ML prob     : {targets['purchase_probability'].mean():.2%}")
        print(f"  Avg hybrid score: {targets['hybrid_score'].mean():.2%}")
        print("\n  Sample rows:")
        print(targets.head(5)[[
            "customer_id", "purchase_probability", "cf_score", "hybrid_score"
        ]].to_string(index=False))

    print("\n" + "=" * 70)
    print(" DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
