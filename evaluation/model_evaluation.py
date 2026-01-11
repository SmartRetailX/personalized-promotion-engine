"""
Model Evaluation and Comparison
Generates comprehensive evaluation metrics for research paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, precision_recall_curve,
    confusion_matrix
)
import os


class ModelEvaluator:
    """
    Comprehensive evaluation of promotion targeting models
    Generates metrics for research paper
    """
    
    def __init__(self, results_dir='evaluation/results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def evaluate_targeting_accuracy(self, y_true, y_pred, y_scores, k_values=[10, 20, 50, 100]):
        """
        Evaluate targeting accuracy at different K values
        Key metrics: Precision@K, Recall@K, NDCG@K
        """
        results = {}
        
        for k in k_values:
            # Get top K predictions
            top_k_indices = np.argsort(y_scores)[-k:]
            y_pred_k = np.zeros_like(y_pred)
            y_pred_k[top_k_indices] = 1
            
            # Calculate metrics
            precision_k = precision_score(y_true, y_pred_k, zero_division=0)
            recall_k = recall_score(y_true, y_pred_k, zero_division=0)
            f1_k = f1_score(y_true, y_pred_k, zero_division=0)
            
            results[f'precision@{k}'] = precision_k
            results[f'recall@{k}'] = recall_k
            results[f'f1@{k}'] = f1_k
        
        # Overall metrics
        results['roc_auc'] = roc_auc_score(y_true, y_scores)
        results['avg_precision'] = np.mean([results[f'precision@{k}'] for k in k_values])
        results['avg_recall'] = np.mean([results[f'recall@{k}'] for k in k_values])
        
        return results
    
    def calculate_business_metrics(self, campaign_results, product_price, discount_pct):
        """
        Calculate business impact metrics
        """
        # Conversion metrics
        conversion_rate = campaign_results['conversions'] / campaign_results['customers_targeted']
        
        # Revenue metrics
        revenue = campaign_results['conversions'] * product_price
        discount_cost = campaign_results['customers_targeted'] * product_price * (discount_pct / 100) * conversion_rate
        profit = revenue - discount_cost
        
        # ROI
        roi = (profit / discount_cost * 100) if discount_cost > 0 else 0
        
        # Cost per acquisition
        cpa = discount_cost / campaign_results['conversions'] if campaign_results['conversions'] > 0 else 0
        
        return {
            'conversion_rate': conversion_rate,
            'total_revenue': revenue,
            'total_cost': discount_cost,
            'total_profit': profit,
            'roi_percentage': roi,
            'cost_per_acquisition': cpa,
            'customers_targeted': campaign_results['customers_targeted'],
            'conversions': campaign_results['conversions']
        }
    
    def compare_strategies(self, results_dict):
        """
        Compare different targeting strategies
        E.g., ML-only vs CF-only vs Hybrid vs Broadcast
        """
        comparison = pd.DataFrame(results_dict).T
        
        # Calculate relative improvements
        if 'broadcast' in comparison.index:
            baseline = comparison.loc['broadcast']
            
            for strategy in comparison.index:
                if strategy != 'broadcast':
                    comparison.loc[strategy, 'roi_improvement_%'] = (
                        (comparison.loc[strategy, 'roi_percentage'] - baseline['roi_percentage']) / 
                        baseline['roi_percentage'] * 100
                    )
                    comparison.loc[strategy, 'cost_reduction_%'] = (
                        (baseline['total_cost'] - comparison.loc[strategy, 'total_cost']) / 
                        baseline['total_cost'] * 100
                    )
        
        return comparison
    
    def plot_precision_recall_curve(self, y_true, y_scores, save_path=None):
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    def plot_strategy_comparison(self, comparison_df, save_path=None):
        """Plot comparison of different strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ROI comparison
        comparison_df['roi_percentage'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('ROI by Strategy', fontsize=12)
        axes[0, 0].set_ylabel('ROI %')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Conversion rate
        comparison_df['conversion_rate'].plot(kind='bar', ax=axes[0, 1], color='green')
        axes[0, 1].set_title('Conversion Rate by Strategy', fontsize=12)
        axes[0, 1].set_ylabel('Conversion Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cost comparison
        comparison_df['total_cost'].plot(kind='bar', ax=axes[1, 0], color='coral')
        axes[1, 0].set_title('Marketing Cost by Strategy', fontsize=12)
        axes[1, 0].set_ylabel('Cost (Rs.)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Profit comparison
        comparison_df['total_profit'].plot(kind='bar', ax=axes[1, 1], color='purple')
        axes[1, 1].set_title('Profit by Strategy', fontsize=12)
        axes[1, 1].set_ylabel('Profit (Rs.)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    def generate_report(self, evaluation_results, save_path=None):
        """
        Generate comprehensive evaluation report for research paper
        """
        report = []
        report.append("=" * 70)
        report.append(" MODEL EVALUATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Targeting Accuracy
        report.append("1. TARGETING ACCURACY METRICS")
        report.append("-" * 70)
        
        if 'targeting_accuracy' in evaluation_results:
            acc = evaluation_results['targeting_accuracy']
            report.append(f"  ROC AUC Score: {acc['roc_auc']:.4f}")
            report.append(f"  Average Precision: {acc['avg_precision']:.4f}")
            report.append(f"  Average Recall: {acc['avg_recall']:.4f}")
            report.append("")
            report.append("  Precision@K:")
            for k in [10, 20, 50, 100]:
                if f'precision@{k}' in acc:
                    report.append(f"    @{k}: {acc[f'precision@{k}']:.4f}")
        
        # Business Impact
        report.append("")
        report.append("2. BUSINESS IMPACT METRICS")
        report.append("-" * 70)
        
        if 'business_metrics' in evaluation_results:
            bm = evaluation_results['business_metrics']
            report.append(f"  Conversion Rate: {bm['conversion_rate']:.2%}")
            report.append(f"  Total Revenue: Rs. {bm['total_revenue']:,.2f}")
            report.append(f"  Total Cost: Rs. {bm['total_cost']:,.2f}")
            report.append(f"  Total Profit: Rs. {bm['total_profit']:,.2f}")
            report.append(f"  ROI: {bm['roi_percentage']:.1f}%")
            report.append(f"  Cost per Acquisition: Rs. {bm['cost_per_acquisition']:,.2f}")
        
        # Strategy Comparison
        report.append("")
        report.append("3. STRATEGY COMPARISON")
        report.append("-" * 70)
        
        if 'strategy_comparison' in evaluation_results:
            comp = evaluation_results['strategy_comparison']
            report.append(comp.to_string())
        
        report.append("")
        report.append("=" * 70)
        
        full_report = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_report)
            print(f"\n✓ Report saved: {save_path}")
        
        print("\n" + full_report)
        
        return full_report


def main():
    """Example evaluation"""
    
    print("=" * 70)
    print(" MODEL EVALUATION - DEMO")
    print("=" * 70)
    
    evaluator = ModelEvaluator()
    
    # Simulate some results
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate predictions
    y_true = np.random.binomial(1, 0.15, n_samples)  # 15% actual purchase rate
    y_scores = np.random.beta(2, 5, n_samples)  # Predicted probabilities
    y_pred = (y_scores > 0.5).astype(int)
    
    # Evaluate targeting accuracy
    print("\nEvaluating targeting accuracy...")
    targeting_metrics = evaluator.evaluate_targeting_accuracy(
        y_true, y_pred, y_scores
    )
    
    # Simulate business metrics
    print("Calculating business metrics...")
    campaign_results = {
        'customers_targeted': 100,
        'conversions': 25
    }
    
    business_metrics = evaluator.calculate_business_metrics(
        campaign_results, product_price=500, discount_pct=15
    )
    
    # Compare strategies
    print("Comparing strategies...")
    strategies_results = {
        'personalized': {
            'roi_percentage': 150,
            'conversion_rate': 0.25,
            'total_cost': 1875,
            'total_profit': 10625
        },
        'broadcast': {
            'roi_percentage': 50,
            'conversion_rate': 0.10,
            'total_cost': 7500,
            'total_profit': 5000
        }
    }
    
    comparison = evaluator.compare_strategies(strategies_results)
    
    # Generate report
    print("\nGenerating evaluation report...")
    evaluation_results = {
        'targeting_accuracy': targeting_metrics,
        'business_metrics': business_metrics,
        'strategy_comparison': comparison
    }
    
    report = evaluator.generate_report(
        evaluation_results,
        save_path='evaluation/results/evaluation_report.txt'
    )
    
    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    os.makedirs('evaluation/results', exist_ok=True)
    main()
