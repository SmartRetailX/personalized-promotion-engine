"""
DB-Aligned Training Script
Trains both Purchase-Prediction (ML) and Collaborative-Filtering models
using data that matches the Prisma schema.

Run from project root:
    python train_db_aligned.py

Models saved to:
    models/db_purchase_prediction_model.pkl
    models/db_collaborative_filtering_model.pkl
"""

import os
import sys
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_analysis.db_preprocessing import DBAlignedPreprocessor
from models.purchase_prediction import PurchasePredictionModel
from models.collaborative_filtering import CollaborativeFilteringModel


def train_purchase_prediction(preprocessor):
    """Train the ML purchase-prediction model."""
    print("\n" + "=" * 70)
    print(" STEP A - PURCHASE PREDICTION MODEL (DB-ALIGNED)")
    print("=" * 70)

    print("\n[1/5] Building time-aware training data ...")
    training_data = preprocessor.create_time_aware_training_data(
        observation_days=90,
        prediction_days=14,
    )

    if len(training_data) == 0:
        print("ERROR: No training data - check that raw_db/ CSVs are present.")
        return None

    print("\n[2/5] Train / Val / Test split ...")
    train_df, val_df, test_df = preprocessor.get_train_val_test_split(
        training_data, val_ratio=0.15, test_ratio=0.15
    )

    model = PurchasePredictionModel(model_type="random_forest")

    print("\n[3/5] Preparing features ...")
    X_train, y_train, _ = model.prepare_features(train_df)
    X_val,   y_val,   _ = model.prepare_features(val_df)
    X_test,  y_test,  _ = model.prepare_features(test_df)

    print(f"  Features    : {list(X_train.columns)}")
    print(f"  Train size  : {len(X_train):,}  (+rate {y_train.mean()*100:.2f}%)")
    print(f"  Val size    : {len(X_val):,}")
    print(f"  Test size   : {len(X_test):,}")

    print("\n[4/5] Training ...")
    model.train(X_train, y_train)

    print("\n[5/5] Threshold tuning on validation set ...")
    model.find_optimal_threshold(X_val, y_val)

    print("\nFinal evaluation on test set (unbiased) ...")
    metrics = model.evaluate(X_test, y_test)

    save_path = os.path.join("models", "db_purchase_prediction_model.pkl")
    model.save_model(save_path)

    if model.feature_importance is not None:
        print("\nFeature importance:")
        print(model.feature_importance.to_string(index=False))

    print(f"\nROC AUC : {metrics['roc_auc']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Threshold: {metrics['optimal_threshold']:.4f}")
    return model


def train_collaborative_filtering(preprocessor):
    """Train the CF similarity model."""
    print("\n" + "=" * 70)
    print(" STEP B - COLLABORATIVE FILTERING MODEL (DB-ALIGNED)")
    print("=" * 70)

    transactions = preprocessor.transactions.copy().sort_values("TransactionDate")

    # 75/25 time split
    split_date = transactions["TransactionDate"].quantile(0.75)
    print(f"\n  Train : up to {split_date.date()}")
    print(f"  Test  : from {split_date.date()}")

    print("\n[1/4] Creating interaction matrix (time-aware) ...")
    train_interactions = preprocessor.create_customer_product_matrix(end_date=split_date)

    test_trans = transactions[transactions["TransactionDate"] >= split_date]
    test_interactions = (
        test_trans.groupby(["CustomerID", "ProductID"])
        .size()
        .reset_index(name="count")
    )

    print("\n[2/4] Training user + item similarity models ...")
    cf_model = CollaborativeFilteringModel(
        n_neighbors_user=50,
        n_neighbors_item=30,
        shrinkage_factor=5,
    )
    cf_model.create_interaction_matrix(train_interactions)
    cf_model.train_user_similarity()
    cf_model.train_item_similarity()

    print("\n[3/4] Evaluating recommendations (Precision@10, Recall@10) ...")
    cf_metrics = cf_model.evaluate_recommendations(test_interactions, k=10)

    print("\n[4/4] Saving model ...")
    save_path = os.path.join("models", "db_collaborative_filtering_model.pkl")
    cf_model.save_model(save_path)

    print(f"\nPrecision@10: {cf_metrics['Precision@K']:.4f}")
    print(f"Recall@10   : {cf_metrics['Recall@K']:.4f}")
    return cf_model


def main():
    print("=" * 70)
    print(" DB-ALIGNED TRAINING PIPELINE")
    print(" Data source : data/raw_db/  (matches Prisma schema)")
    print("=" * 70)

    preprocessor = DBAlignedPreprocessor()
    preprocessor.load_data()

    ml_model = train_purchase_prediction(preprocessor)
    cf_model = train_collaborative_filtering(preprocessor)

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print("  models/db_purchase_prediction_model.pkl")
    print("  models/db_collaborative_filtering_model.pkl")
    print("\nNext step: run  python models/db_promotion_engine.py")


if __name__ == "__main__":
    main()
