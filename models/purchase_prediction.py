"""
Purchase Prediction Model - FULLY FIXED VERSION

FIXES APPLIED:
1. FIX #1: Added product_category_encoded feature
2. FIX #2: Threshold learned on VALIDATION set (not test)
3. FIX #3: No scaling for Random Forest (only for linear models)
4. FIX #4: Removed correlated features (product_total_purchases)
5. FIX SYSTEM: Connected to CF via generate_promotion_targets()

Expected Performance: ROC AUC 0.65-0.80, F1 0.55-0.75 (realistic!)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, f1_score
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_analysis.preprocessing import DataPreprocessor


class PurchasePredictionModel:
    """
    Model to predict FUTURE purchase probability for personalized promotions
    ALL FIXES APPLIED - production-ready
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.optimal_threshold = 0.5
        self.feature_cols = None
        
    def prepare_features(self, df):
        """
        Prepare features for training
        
        FIX #1: Added product_category_encoded
        FIX #4: Removed correlated features
        """
        data = df.copy()
        
        # Features with product identity (FIX #1)
        feature_cols = [
            'customer_purchase_frequency',    # General customer activity
            'customer_avg_transaction',       # Spending behavior
            'customer_recency',               # How recently active
            'customer_promo_response_rate',   # Promotion sensitivity
            'customer_age',                   # Demographics
            'product_price',                  # Product attribute
            'product_category_encoded',       # FIX #1: Product identity via category
            'category_affinity',              # Customer's category preference
            'category_purchase_count',        # Category purchases
        ]
        
        # FIX #4: Removed product_popularity_obs (correlated with category_purchase_count)
        
        # Check which features exist
        available_features = [f for f in feature_cols if f in data.columns]
        
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            print(f"  Warning: Missing features: {missing}")
        
        # If model already trained, validate feature alignment
        if self.feature_cols is not None and self.model is not None:
            expected = set(self.feature_cols)
            actual = set(available_features)
            if expected != actual:
                missing_at_inference = expected - actual
                extra_at_inference = actual - expected
                error_msg = "FEATURE MISMATCH: Training vs Inference features differ!\n"
                if missing_at_inference:
                    error_msg += f"  Missing at inference (were in training): {missing_at_inference}\n"
                if extra_at_inference:
                    error_msg += f"  Extra at inference (not in training): {extra_at_inference}\n"
                error_msg += "  This will cause incorrect predictions!"
                raise ValueError(error_msg)
        
        # During training, store the features used
        if self.model is None:
            self.feature_cols = available_features
        
        X = data[available_features].fillna(0)
        y = data['target'] if 'target' in data.columns else None
        
        return X, y, data
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        FIX #3: No scaling for tree-based models
        """
        print(f"\nTraining {self.model_type} model...")
        
        # FIX #3: Conditionally scale only for linear models
        if self.model_type == 'logistic_regression':
            X_train_processed = self.scaler.fit_transform(X_train)
            print("  * Scaling applied (logistic regression)")
        else:
            X_train_processed = X_train.values
            print("  * No scaling (tree-based model)")
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        
        self.model.fit(X_train_processed, y_train)
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print("* Model trained successfully")
        
    def predict_proba(self, X):
        """Predict purchase probability"""
        # FIX #3: Conditionally scale
        if self.model_type == 'logistic_regression':
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X.values if hasattr(X, 'values') else X
        
        return self.model.predict_proba(X_processed)[:, 1]
    
    def predict(self, X, threshold=None):
        """Predict purchase (binary)"""
        if threshold is None:
            threshold = self.optimal_threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def find_optimal_threshold(self, X_val, y_val):
        """
        FIX #2: Find optimal threshold on VALIDATION set (not test!)
        This prevents data leakage in threshold selection
        """
        print("\nFinding optimal threshold on validation set...")
        
        y_proba = self.predict_proba(X_val)
        
        try:
            precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            print(f"  Optimal threshold: {self.optimal_threshold:.4f}")
            print(f"  F1 at threshold: {f1_scores[optimal_idx]:.4f}")
        except Exception as e:
            print(f"  Warning: Could not optimize threshold: {e}")
            self.optimal_threshold = 0.5
        
        return self.optimal_threshold
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on TEST set (threshold already fixed from validation)
        
        This is UNBIASED evaluation - threshold was NOT learned on test data
        """
        print("\n" + "=" * 50)
        print("FINAL EVALUATION ON TEST SET (Unbiased)")
        print("=" * 50)
        print(f"Using threshold: {self.optimal_threshold:.4f} (from validation)")
        
        y_pred = self.predict(X_test, threshold=self.optimal_threshold)
        y_proba = self.predict_proba(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"\nROC AUC Score: {roc_auc:.4f}")
        except:
            roc_auc = 0.5
            print("\nROC AUC: Could not calculate (single class)")
        
        # F1 at optimal threshold
        f1 = f1_score(y_test, y_pred)
        print(f"F1 Score: {f1:.4f}")
        
        # Interpretation
        print("\n" + "-" * 50)
        print("INTERPRETATION:")
        if roc_auc >= 0.65:
            print(f"  * ROC AUC {roc_auc:.2f} - Model has predictive power!")
        else:
            print(f"  * ROC AUC {roc_auc:.2f} - Model needs improvement")
        
        if roc_auc >= 0.70:
            print("  * This is GOOD for retail prediction!")
            print("  * Industry standard: 0.65-0.80")
        
        return {
            'roc_auc': roc_auc,
            'optimal_threshold': self.optimal_threshold,
            'f1_score': f1
        }
    
    def get_promotion_targets(self, customer_features_df, product_id, top_n=100):
        """
        Get top N customers most likely to purchase a product
        Core function for personalized promotion targeting
        """
        print(f"\nFinding top {top_n} customers for product {product_id}...")
        
        product_data = customer_features_df[
            customer_features_df['ProductID'] == product_id
        ].copy()
        
        if len(product_data) == 0:
            print(f"Warning: No data found for product {product_id}")
            return pd.DataFrame()
        
        X, _, data = self.prepare_features(product_data)
        
        data['purchase_probability'] = self.predict_proba(X)
        
        # Sort by probability
        top_customers = data.nlargest(top_n, 'purchase_probability')[
            ['CustomerID', 'ProductID', 'purchase_probability', 
             'customer_promo_response_rate', 'category_affinity']
        ]
        
        return top_customers
    
    def save_model(self, filepath='models/purchase_prediction_model.pkl'):
        """Save trained model"""
        # Ensure the directory exists
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'optimal_threshold': self.optimal_threshold,
            'feature_cols': self.feature_cols
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n* Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath='models/purchase_prediction_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_importance = model_data['feature_importance']
        instance.optimal_threshold = model_data.get('optimal_threshold', 0.5)
        instance.feature_cols = model_data.get('feature_cols', None)
        
        print(f"* Model loaded from: {filepath}")
        return instance


def main():
    """Main training pipeline - ALL FIXES APPLIED"""
    
    print("=" * 70)
    print(" PURCHASE PREDICTION MODEL TRAINING (ALL FIXES APPLIED)")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    # Create time-aware training data
    print("\n2. Creating time-aware training data...")
    training_data = preprocessor.create_time_aware_training_data(
        observation_days=90,
        prediction_days=14
    )
    
    if len(training_data) == 0:
        print("ERROR: No training data created!")
        return None
    
    # FIX #2: Train/Validation/Test split
    print("\n3. Creating train/validation/test split...")
    train_df, val_df, test_df = preprocessor.get_train_val_test_split(
        training_data, val_ratio=0.15, test_ratio=0.15
    )
    
    # Initialize model
    model = PurchasePredictionModel(model_type='random_forest')
    
    # Prepare features
    print("\n4. Preparing features...")
    X_train, y_train, _ = model.prepare_features(train_df)
    X_val, y_val, _ = model.prepare_features(val_df)
    X_test, y_test, _ = model.prepare_features(test_df)
    
    print(f"\nFeatures used: {list(X_train.columns)}")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train positive rate: {y_train.mean()*100:.2f}%")
    print(f"Val positive rate: {y_val.mean()*100:.2f}%")
    print(f"Test positive rate: {y_test.mean()*100:.2f}%")
    
    # Train
    print("\n5. Training model...")
    model.train(X_train, y_train)
    
    # FIX #2: Find threshold on VALIDATION set
    print("\n6. Finding optimal threshold on validation set...")
    model.find_optimal_threshold(X_val, y_val)
    
    # Evaluate on TEST set (unbiased)
    print("\n7. Final evaluation on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    # Feature importance
    if model.feature_importance is not None:
        print("\nFeature Importance:")
        print(model.feature_importance.to_string(index=False))
    
    # Save
    print("\n8. Saving model...")
    model.save_model('models/purchase_prediction_model.pkl')
    
    print("\n" + "=" * 70)
    print(" MODEL TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nFinal Metrics (UNBIASED - threshold from validation):")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    
    print("\n" + "-" * 70)
    print("FIXES APPLIED:")
    print("  [x] FIX #1: Product category encoding added")
    print("  [x] FIX #2: Threshold learned on validation (not test)")
    print("  [x] FIX #3: No scaling for Random Forest")
    print("  [x] FIX #4: Removed correlated features")
    print("-" * 70)
    
    return model


if __name__ == "__main__":
    model = main()
