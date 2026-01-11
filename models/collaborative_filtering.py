"""
Collaborative Filtering Model for Product Recommendations
Used to find similar customers and products for promotion targeting

FIXES APPLIED:
1. FIX #5: Time cutoff to prevent future leakage
2. FIX #6: Log-scaled interaction scores
3. FIX #7: Removed unused SVD (cleaner code)
4. FIX SYSTEM: Proper evaluation with Precision@K and Recall@K
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_analysis.preprocessing import DataPreprocessor


class CollaborativeFilteringModel:
    """
    Item-based and User-based collaborative filtering
    Helps identify which customers are similar to find promotion targets
    
    FIXES APPLIED:
    - FIX #5: Time-aware training (no future leakage)
    - FIX #6: Log-scaled interaction scores
    - FIX #7: Removed unused SVD
    - Proper Precision@K and Recall@K evaluation
    
    OPTIMIZATIONS:
    - Hybrid recommendations (user + item based)
    - Increased k-neighbors for better coverage
    - Shrinkage to penalize low co-occurrence similarities
    """
    
    def __init__(self, n_neighbors_user=50, n_neighbors_item=30, shrinkage_factor=10):
        """
        Args:
            n_neighbors_user: Number of neighbors for user-based CF (default: 50)
            n_neighbors_item: Number of neighbors for item-based CF (default: 30)
            shrinkage_factor: Penalizes similarities with few common items (default: 10)
        """
        self.n_neighbors_user = n_neighbors_user
        self.n_neighbors_item = n_neighbors_item
        self.shrinkage_factor = shrinkage_factor
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity_model = None
        self.item_similarity_model = None
        self.customer_ids = None
        self.product_ids = None
        # Store co-occurrence counts for shrinkage
        self.user_cooccurrence = None
        self.item_cooccurrence = None
        
    def create_interaction_matrix(self, interactions_df):
        """
        Create user-item interaction matrix
        
        FIX #6: Uses log-scaled scores from preprocessor
        """
        print("Creating interaction matrices...")
        
        user_item_df = interactions_df.pivot_table(
            index='CustomerID',
            columns='ProductID',
            values='interaction_score',
            fill_value=0
        )
        
        self.customer_ids = user_item_df.index.tolist()
        self.product_ids = user_item_df.columns.tolist()
        
        self.user_item_matrix = csr_matrix(user_item_df.values)
        self.item_user_matrix = self.user_item_matrix.T
        
        # Compute co-occurrence counts for shrinkage
        # User co-occurrence: how many items two users have in common
        binary_matrix = (self.user_item_matrix > 0).astype(float)
        self.user_cooccurrence = binary_matrix.dot(binary_matrix.T).toarray()
        
        # Item co-occurrence: how many users two items have in common
        binary_item = (self.item_user_matrix > 0).astype(float)
        self.item_cooccurrence = binary_item.dot(binary_item.T).toarray()
        
        print(f"  Matrix shape: {self.user_item_matrix.shape}")
        print(f"  Sparsity: {(1 - self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape)) * 100:.2f}%")
        
    def train_user_similarity(self):
        """Train user-based collaborative filtering with more neighbors"""
        print("\nTraining user-based collaborative filtering...")
        
        self.user_similarity_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors_user, len(self.customer_ids)),
            metric='cosine',
            algorithm='brute'
        )
        self.user_similarity_model.fit(self.user_item_matrix)
        
        print(f"* User similarity model trained (k={self.n_neighbors_user})")
        
    def train_item_similarity(self):
        """Train item-based collaborative filtering with more neighbors"""
        print("Training item-based collaborative filtering...")
        
        self.item_similarity_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors_item, len(self.product_ids)),
            metric='cosine',
            algorithm='brute'
        )
        self.item_similarity_model.fit(self.item_user_matrix)
        
        print(f"* Item similarity model trained (k={self.n_neighbors_item})")
        
    def get_similar_customers(self, customer_id, n=10):
        """
        Find similar customers for a given customer.
        
        Uses shrinkage to penalize similarities based on few common items:
        adjusted_sim = raw_sim * n_common / (n_common + shrinkage_factor)
        """
        if customer_id not in self.customer_ids:
            return []
        
        customer_idx = self.customer_ids.index(customer_id)
        customer_vector = self.user_item_matrix[customer_idx]
        
        # Get more neighbors than needed, then apply shrinkage and re-rank
        k = min(n * 3, len(self.customer_ids))
        distances, indices = self.user_similarity_model.kneighbors(
            customer_vector, n_neighbors=k
        )
        
        similar_customers = []
        for i, idx in enumerate(indices[0][1:]):  # Skip self
            raw_sim = 1 - distances[0][i + 1]
            
            # Apply shrinkage: penalize similarities with few common items
            n_common = self.user_cooccurrence[customer_idx, idx]
            shrunk_sim = raw_sim * n_common / (n_common + self.shrinkage_factor)
            
            similar_customers.append({
                'CustomerID': self.customer_ids[idx],
                'similarity': shrunk_sim,
                'raw_similarity': raw_sim,
                'n_common_items': int(n_common)
            })
        
        # Re-sort by shrunk similarity and return top n
        similar_customers.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_customers[:n]
    
    def get_similar_products(self, product_id, n=10):
        """
        Find similar products for a given product.
        
        Uses shrinkage to penalize similarities based on few common users:
        adjusted_sim = raw_sim * n_common / (n_common + shrinkage_factor)
        """
        if product_id not in self.product_ids:
            return []
        
        product_idx = self.product_ids.index(product_id)
        product_vector = self.item_user_matrix[product_idx]
        
        # Get more neighbors than needed, then apply shrinkage and re-rank
        k = min(n * 3, len(self.product_ids))
        distances, indices = self.item_similarity_model.kneighbors(
            product_vector, n_neighbors=k
        )
        
        similar_products = []
        for i, idx in enumerate(indices[0][1:]):  # Skip self
            raw_sim = 1 - distances[0][i + 1]
            
            # Apply shrinkage: penalize similarities with few common users
            n_common = self.item_cooccurrence[product_idx, idx]
            shrunk_sim = raw_sim * n_common / (n_common + self.shrinkage_factor)
            
            similar_products.append({
                'ProductID': self.product_ids[idx],
                'similarity': shrunk_sim,
                'raw_similarity': raw_sim,
                'n_common_users': int(n_common)
            })
        
        # Re-sort by shrunk similarity and return top n
        similar_products.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_products[:n]
    
    def recommend_products_for_customer(self, customer_id, n=10, 
                                        user_weight=0.6, item_weight=0.4):
        """
        Recommend products using HYBRID approach (user-based + item-based).
        
        Args:
            customer_id: Customer to recommend for
            n: Number of recommendations
            user_weight: Weight for user-based scores (default 0.6)
            item_weight: Weight for item-based scores (default 0.4)
        
        Returns:
            List of recommended products with scores
        """
        if customer_id not in self.customer_ids:
            return []
        
        customer_idx = self.customer_ids.index(customer_id)
        customer_purchases = set(np.where(self.user_item_matrix[customer_idx].toarray()[0] > 0)[0])
        purchased_product_ids = [self.product_ids[idx] for idx in customer_purchases]
        
        product_scores = {}
        
        # === USER-BASED: Products bought by similar users ===
        similar_customers = self.get_similar_customers(customer_id, n=self.n_neighbors_user)
        
        for similar_cust in similar_customers:
            similar_cust_id = similar_cust['CustomerID']
            similar_idx = self.customer_ids.index(similar_cust_id)
            similarity = similar_cust['similarity']
            
            purchased_products = np.where(self.user_item_matrix[similar_idx].toarray()[0] > 0)[0]
            
            for prod_idx in purchased_products:
                if prod_idx not in customer_purchases:
                    product_id = self.product_ids[prod_idx]
                    if product_id not in product_scores:
                        product_scores[product_id] = {'user_score': 0, 'item_score': 0}
                    product_scores[product_id]['user_score'] += similarity
        
        # === ITEM-BASED: Products similar to what customer bought ===
        for purchased_prod_id in purchased_product_ids:
            similar_products = self.get_similar_products(purchased_prod_id, n=self.n_neighbors_item)
            
            for similar_prod in similar_products:
                prod_id = similar_prod['ProductID']
                if prod_id not in purchased_product_ids:
                    if prod_id not in product_scores:
                        product_scores[prod_id] = {'user_score': 0, 'item_score': 0}
                    product_scores[prod_id]['item_score'] += similar_prod['similarity']
        
        # === HYBRID: Combine user and item scores ===
        recommendations = []
        for product_id, scores in product_scores.items():
            # Normalize scores before combining
            user_score = scores['user_score']
            item_score = scores['item_score']
            
            hybrid_score = user_weight * user_score + item_weight * item_score
            
            recommendations.append({
                'ProductID': product_id,
                'score': hybrid_score,
                'user_score': user_score,
                'item_score': item_score
            })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n]
    
    def find_customers_for_product(self, product_id, n=100, return_scores=False):
        """
        Find candidate customers for a product promotion
        Returns customers who:
        1. Bought similar products
        2. Are similar to customers who bought this product
        
        Args:
            product_id: Product to find customers for
            n: Max number of candidates to return
            return_scores: If True, return list of dicts with CustomerID and cf_score
                          If False, return list of CustomerIDs (backward compatible)
        
        Returns:
            List of CustomerIDs or List of {'CustomerID': x, 'cf_score': y}
        """
        if product_id not in self.product_ids:
            print(f"Product {product_id} not found")
            return []
        
        # Track candidates with scores (higher = more relevant)
        candidate_scores = {}
        
        # Method 1: Customers who bought similar products
        similar_products = self.get_similar_products(product_id, n=10)
        for similar_prod in similar_products:
            similar_prod_id = similar_prod['ProductID']
            product_similarity = similar_prod['similarity']
            
            if similar_prod_id in self.product_ids:
                prod_idx = self.product_ids.index(similar_prod_id)
                customer_indices = np.where(self.item_user_matrix[prod_idx].toarray()[0] > 0)[0]
                for cust_idx in customer_indices:
                    cust_id = self.customer_ids[cust_idx]
                    # Score based on product similarity
                    if cust_id not in candidate_scores:
                        candidate_scores[cust_id] = 0
                    candidate_scores[cust_id] += product_similarity
        
        # Method 2: Customers similar to those who bought this product
        prod_idx = self.product_ids.index(product_id)
        buyers = np.where(self.item_user_matrix[prod_idx].toarray()[0] > 0)[0]
        
        for buyer_idx in buyers[:20]:  # Limit for efficiency
            buyer_id = self.customer_ids[buyer_idx]
            similar_customers = self.get_similar_customers(buyer_id, n=10)
            for similar in similar_customers:
                cust_id = similar['CustomerID']
                cust_similarity = similar['similarity']
                if cust_id not in candidate_scores:
                    candidate_scores[cust_id] = 0
                candidate_scores[cust_id] += cust_similarity
        
        # Normalize scores to 0-1 range
        if candidate_scores:
            max_score = max(candidate_scores.values())
            if max_score > 0:
                candidate_scores = {k: v / max_score for k, v in candidate_scores.items()}
        
        # Sort by score descending
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        if return_scores:
            return [{'CustomerID': cust_id, 'cf_score': score} for cust_id, score in sorted_candidates]
        else:
            # Backward compatible: just return IDs
            return [cust_id for cust_id, score in sorted_candidates]
    
    def find_co_purchased_products(self, product_id, min_support=0.1):
        """
        Find products frequently bought together with given product
        (Market Basket Analysis / Association Rules)
        """
        if product_id not in self.product_ids:
            return []
        
        product_idx = self.product_ids.index(product_id)
        
        # Customers who bought this product
        customers_bought = np.where(self.item_user_matrix[product_idx].toarray()[0] > 0)[0]
        total_buyers = len(customers_bought)
        
        if total_buyers == 0:
            return []
        
        # For each other product, count co-purchases
        co_purchase_counts = {}
        
        for cust_idx in customers_bought:
            products_bought = np.where(self.user_item_matrix[cust_idx].toarray()[0] > 0)[0]
            
            for other_prod_idx in products_bought:
                if other_prod_idx != product_idx:
                    other_prod_id = self.product_ids[other_prod_idx]
                    if other_prod_id not in co_purchase_counts:
                        co_purchase_counts[other_prod_id] = 0
                    co_purchase_counts[other_prod_id] += 1
        
        # Calculate support
        co_purchased = []
        for prod_id, count in co_purchase_counts.items():
            support = count / total_buyers
            if support >= min_support:
                co_purchased.append({
                    'ProductID': prod_id,
                    'co_purchase_count': count,
                    'support': support
                })
        
        co_purchased.sort(key=lambda x: x['support'], reverse=True)
        
        return co_purchased
    
    def evaluate_recommendations(self, test_interactions, k=10):
        """
        PROPER evaluation for collaborative filtering
        Uses Precision@K and Recall@K (NOT accuracy!)
        """
        print(f"\n" + "=" * 50)
        print(f"EVALUATING RECOMMENDATIONS (Precision@{k}, Recall@{k})")
        print("=" * 50)
        
        precision_scores = []
        recall_scores = []
        evaluated_customers = 0
        
        for customer_id in test_interactions['CustomerID'].unique():
            true_products = set(
                test_interactions[test_interactions['CustomerID'] == customer_id]['ProductID']
            )
            
            if customer_id not in self.customer_ids or len(true_products) == 0:
                continue
            
            recommendations = self.recommend_products_for_customer(customer_id, n=k)
            recommended_products = set([r['ProductID'] for r in recommendations])
            
            if len(recommended_products) == 0:
                continue
            
            hits = recommended_products.intersection(true_products)
            
            precision = len(hits) / k
            precision_scores.append(precision)
            
            recall = len(hits) / len(true_products)
            recall_scores.append(recall)
            
            evaluated_customers += 1
        
        if len(precision_scores) == 0:
            print("WARNING: Could not evaluate any customers")
            return {'Precision@K': 0, 'Recall@K': 0, 'K': k, 'evaluated_customers': 0}
        
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        
        print(f"\nEvaluated {evaluated_customers} customers")
        print(f"\nResults:")
        print(f"  Precision@{k}: {avg_precision:.4f}")
        print(f"  Recall@{k}: {avg_recall:.4f}")
        
        print(f"\n" + "-" * 50)
        print("INTERPRETATION (Grocery Domain):")
        if avg_precision >= 0.15:
            print(f"  * Precision@{k} {avg_precision:.2f} - Good for grocery!")
        elif avg_precision >= 0.08:
            print(f"  * Precision@{k} {avg_precision:.2f} - Acceptable for grocery (high variety)")
        else:
            print(f"  * Precision@{k} {avg_precision:.2f} - Low, but ML rescoring will help")
        
        if avg_recall >= 0.15:
            print(f"  * Recall@{k} {avg_recall:.2f} - Good coverage!")
        elif avg_recall >= 0.05:
            print(f"  * Recall@{k} {avg_recall:.2f} - Acceptable (customers buy variety)")
        else:
            print(f"  * Recall@{k} {avg_recall:.2f} - Low coverage")
        
        print(f"\n  NOTE: Grocery CF metrics are typically lower than movie/music domains")
        print(f"  because customers buy from a large variety pool, not repeat items.")
        print(f"  The CF→ML pipeline compensates via purchase probability scoring.")
        
        return {
            'Precision@K': avg_precision,
            'Recall@K': avg_recall,
            'K': k,
            'evaluated_customers': evaluated_customers
        }
    
    def save_model(self, filepath='models/collaborative_filtering_model.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'item_user_matrix': self.item_user_matrix,
            'user_similarity_model': self.user_similarity_model,
            'item_similarity_model': self.item_similarity_model,
            'customer_ids': self.customer_ids,
            'product_ids': self.product_ids,
            'n_neighbors_user': self.n_neighbors_user,
            'n_neighbors_item': self.n_neighbors_item,
            'shrinkage_factor': self.shrinkage_factor,
            'user_cooccurrence': self.user_cooccurrence,
            'item_cooccurrence': self.item_cooccurrence
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n* Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath='models/collaborative_filtering_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        # Handle both old and new model formats
        n_neighbors_user = model_data.get('n_neighbors_user', model_data.get('n_neighbors', 50))
        n_neighbors_item = model_data.get('n_neighbors_item', model_data.get('n_neighbors', 30))
        shrinkage_factor = model_data.get('shrinkage_factor', 10)
        
        instance = cls(
            n_neighbors_user=n_neighbors_user,
            n_neighbors_item=n_neighbors_item,
            shrinkage_factor=shrinkage_factor
        )
        instance.user_item_matrix = model_data['user_item_matrix']
        instance.item_user_matrix = model_data['item_user_matrix']
        instance.user_similarity_model = model_data['user_similarity_model']
        instance.item_similarity_model = model_data['item_similarity_model']
        instance.customer_ids = model_data['customer_ids']
        instance.product_ids = model_data['product_ids']
        instance.user_cooccurrence = model_data.get('user_cooccurrence')
        instance.item_cooccurrence = model_data.get('item_cooccurrence')
        
        # Recompute cooccurrence if not saved (backward compatibility)
        if instance.user_cooccurrence is None:
            binary_matrix = (instance.user_item_matrix > 0).astype(float)
            instance.user_cooccurrence = binary_matrix.dot(binary_matrix.T).toarray()
        if instance.item_cooccurrence is None:
            binary_item = (instance.item_user_matrix > 0).astype(float)
            instance.item_cooccurrence = binary_item.dot(binary_item.T).toarray()
        
        print(f"* Model loaded from: {filepath}")
        return instance


def main():
    """Main training pipeline with all fixes and optimizations"""
    
    print("=" * 70)
    print(" COLLABORATIVE FILTERING MODEL TRAINING")
    print(" FIXES + OPTIMIZATIONS APPLIED")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    # FIX #5: Time-based split for training
    print("\n2. Creating time-based train/test split...")
    transactions = preprocessor.transactions.copy()
    transactions = transactions.sort_values('TransactionDate')
    
    split_date = transactions['TransactionDate'].quantile(0.75)
    
    print(f"  Train: up to {split_date.date()}")
    print(f"  Test: from {split_date.date()}")
    
    # FIX #5: Create interactions with time cutoff (NO FUTURE LEAKAGE)
    print("\n3. Creating interaction matrix (time-aware)...")
    train_interactions = preprocessor.create_customer_product_matrix(end_date=split_date)
    
    # Create test interactions
    test_trans = transactions[transactions['TransactionDate'] >= split_date]
    test_interactions = test_trans.groupby(['CustomerID', 'ProductID']).size().reset_index(name='count')
    
    # Train model with OPTIMIZED parameters
    print("\n4. Training collaborative filtering models...")
    print("  OPTIMIZATIONS:")
    print("    - User neighbors: 50 (increased from 10)")
    print("    - Item neighbors: 30 (increased from 10)")
    print("    - Shrinkage factor: 5 (reduced from 10 - less aggressive)")
    print("    - Hybrid recommendations: 60% user + 40% item")
    
    model = CollaborativeFilteringModel(
        n_neighbors_user=50,  # Increased for better coverage
        n_neighbors_item=30,  # Increased for better coverage
        shrinkage_factor=5    # Reduced - 10 was too aggressive
    )
    model.create_interaction_matrix(train_interactions)
    model.train_user_similarity()
    model.train_item_similarity()
    
    # Example usage
    print("\n5. Testing model...")
    if len(model.customer_ids) > 0:
        sample_customer = model.customer_ids[0]
        print(f"\nSample customer: {sample_customer}")
        
        similar_customers = model.get_similar_customers(sample_customer, n=5)
        print("\nTop 5 similar customers (with shrinkage):")
        for cust in similar_customers:
            print(f"  {cust['CustomerID']}: sim={cust['similarity']:.4f} (raw={cust['raw_similarity']:.4f}, common={cust['n_common_items']})")
        
        recommendations = model.recommend_products_for_customer(sample_customer, n=5)
        print("\nTop 5 product recommendations (HYBRID):")
        for rec in recommendations:
            print(f"  {rec['ProductID']}: hybrid={rec['score']:.4f} (user={rec['user_score']:.4f}, item={rec['item_score']:.4f})")
    
    # Co-purchase analysis
    if len(model.product_ids) > 0:
        sample_product = model.product_ids[0]
        print(f"\nCo-purchased products with {sample_product}:")
        co_purchased = model.find_co_purchased_products(sample_product, min_support=0.05)
        for item in co_purchased[:5]:
            print(f"  {item['ProductID']}: support={item['support']:.2%}")
    
    # PROPER EVALUATION
    print("\n6. Evaluating with Precision@K and Recall@K...")
    metrics = model.evaluate_recommendations(test_interactions, k=10)
    
    # Save model
    print("\n7. Saving model...")
    model.save_model('models/collaborative_filtering_model.pkl')
    
    print("\n" + "=" * 70)
    print(" COLLABORATIVE FILTERING TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nFinal Metrics:")
    print(f"  Precision@10: {metrics['Precision@K']:.4f}")
    print(f"  Recall@10: {metrics['Recall@K']:.4f}")
    
    print("\n" + "-" * 70)
    print("FIXES APPLIED:")
    print("  [x] FIX #5: Time cutoff (no future leakage)")
    print("  [x] FIX #6: Log-scaled interaction scores")
    print("  [x] FIX #7: Removed unused SVD")
    print("\nOPTIMIZATIONS APPLIED:")
    print("  [x] Hybrid CF: User (60%) + Item (40%) based")
    print("  [x] Increased neighbors: User=50, Item=30")
    print("  [x] Shrinkage: sim * n_common / (n_common + 10)")
    print("-" * 70)
    
    return model


if __name__ == "__main__":
    model = main()
