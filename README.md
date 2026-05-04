**Personalized Promotion Engine** - An AI system that predicts which customers will buy specific products and generates targeted promotion campaigns.

## 💡 What happens in this promotion engine

**Traditional Way (Many supermarket systems current approach):**

```
Bread is 15% off
→ Send SMS to ALL 10,000 customers
→ Only 500 buy bread
→ Wasted: 9,500 SMS, discounts to customers who won't buy
```

**Proposed promotion engine AI Way:**

```
Bread is 15% off
→ AI analyzes: Who buys bread regularly?
→ Finds 1,000 most likely customers
→ Only send to these 1,000
→ 400 buy bread (40% conversion vs 5%)
→ Saved: 9,000 SMS, no wasted discounts
```

### How Does My AI Work?

**Step 1: Learn Patterns**

```
AI looks at past data:
- Customer A buys bread every week
- Customer B never bought bread
- Customer C bought bread when discounted
```

**Step 2: Predict Future**

```
New bread promotion:
- Customer A: 90% will buy (high priority)
- Customer B: 5% will buy (skip)
- Customer C: 60% if discount >15% (target with 20% discount)
```

**Step 3: Optimize**

```
 system decides:
- Who to target (top 1000)
- What discount to give each (5-25% personalized)
- When to send (morning vs evening)
- What to bundle (bread + butter)
```

---

## 🎯Research Goals

By end of project, should be able to:

### Demonstrate:

1. **Working System**: Full end-to-end promotion targeting
2. **Better Performance**: Your way beats traditional broadcast
3. **Research Novelty**: Unique combination of techniques

### Explain:

1. **Why**: Why personalization is better
2. **How**: How ML makes predictions
3. **What**: What business value it creates

### Prove:

1. **Accuracy**: Models predict well (metrics)
2. **Impact**: ROI improvement (money saved/earned)
3. **Fairness**: No demographic discrimination

---

# 📁 Complete Project Structure

```
Personalized Promotion Engine/
│
│
├── 📂 data/                              ← All datasets
│   ├── 📂 raw_db/                       ← DB-aligned CSV files
│   │   ├── users.csv
│   │   ├── categories.csv
│   │   ├── products.csv
│   │   ├── promotions.csv
│   │   ├── orders.csv
│   │   └── transactions.csv
│   │
│   └── 📂 processed/                    ← ML-ready features
│       ├── customer_features.csv
│       ├── product_features.csv
│       ├── customer_product_interactions.csv
│       ├── category_spending_pct.csv
│       └── category_purchases.csv
│
├── 📂 data_generation/                   ← Dataset creation scripts
│   └── generate_db_aligned_dataset.py   ← DB-aligned dataset generator
│
├── 📂 data_analysis/                     ← Data processing & features
│   └── db_preprocessing.py              ← DB-aligned feature engineering
│                                         - RFM analysis
│                                         - Interaction matrices
│                                         - Category preferences
│
├── 📂 models/                            ← Machine Learning models
│   ├── purchase_prediction.py           ← [CORE] Predict purchase probability
│   │                                     - Random Forest classifier
│   │                                     - Feature importance
│   │
│   ├── collaborative_filtering.py       ← Find similar customers/products
│   │                                     - User-based CF
│   │                                     - Item-based CF
│   │                                     - Matrix factorization (SVD)
│   │
│   ├── promotion_optimizer.py           ← [ADVANCED] Optimize campaigns
│   │                                     - Personalized discounts
│   │                                     - Fatigue detection
│   │                                     - ROI prediction
│   │
│   ├── causal_inference.py              ← [RESEARCH NOVELTY] Prove impact
│   │                                     - Uplift modeling
│   │                                     - Persuadables identification
│   │                                     - Incremental ROI
│   │
│   ├── db_promotion_engine.py           ← [COMPLETE SYSTEM] DB-aligned engine
│   │                                     - End-to-end campaigns
│   │                                     - Strategy comparison
│   │                                     - Personalized vs broadcast
│   │
│   ├── db_purchase_prediction_model.pkl    ← Trained model (after running)
│   └── db_collaborative_filtering_model.pkl ← Trained model (after running)
│
├── 📂 evaluation/                        ← Model evaluation & metrics
│   ├── model_evaluation.py              ← Comprehensive evaluation
│   │                                     - Precision@K, Recall@K
│   │                                     - Business metrics (ROI, conversion)
│   │                                     - Report generation
│   │
│   └── 📂 results/                      ← Evaluation outputs (after running)
│       ├── evaluation_report.txt
│       ├── precision_recall_curve.png
│       └── strategy_comparison.png
│
└── 📂 notebooks/                         ← Jupyter notebooks (optional)
    ├── 01_exploratory_data_analysis.ipynb
    ├── 02_model_training.ipynb
    └── 03_results_visualization.ipynb
```

## 🚀 Workflow Diagram

```
START
  ↓
1. Setup (ONE TIME)
  ├─ Install Python
  ├─ pip install -r requirements.txt
  └─ python quick_start.py
     ↓
2. Data Generation
  ├─ config.py (settings)
  └─ generate_db_aligned_dataset.py
     │
      ├─→ users.csv
      ├─→ categories.csv
      ├─→ products.csv
      ├─→ promotions.csv
      ├─→ orders.csv
      └─→ transactions.csv
        ↓
3. Data Processing
  └─ db_preprocessing.py
     │
     ├─→ customer_features.csv
     ├─→ product_features.csv
     └─→ interactions.csv
        ↓
4. Model Training
    └─ train_db_aligned.py → db_*_model.pkl
     ↓
5. Advanced Features
  ├─ promotion_optimizer.py
  └─ causal_inference.py
     ↓
6. Integration
  └─ db_promotion_engine.py
     ↓
7. Evaluation
  └─ model_evaluation.py
     │
     ├─→ Reports
     ├─→ Charts
     └─→ Metrics for paper
        ↓
DONE! 🎉
```

### Targeting Accuracy:

- **Precision@100**: ~0.60 (60% of targeted customers buy)
- **Recall@100**: ~0.45 (find 45% of potential buyers)
- **ROC AUC**: ~0.75-0.80 (good discrimination)

## 📌 PROBLEM SOLVED

**Before (Traditional):**

```
Store wants to promote Bread
    ↓
Send 10% discount email to ALL 1000 customers
    ↓
Only 100 customers buy (10% conversion)
    ↓
Wasted 900 emails, high marketing cost
```

**After (My AI Solution):**

```
Store wants to promote Bread
    ↓
AI analyzes customer data → Predicts who will buy
    ↓
Send 10% discount email to only 50 targeted customers
    ↓
12-13 customers buy (25% conversion)
    ↓
Same sales, 75% cost reduction!
```

### Model Training (DONE ✅)

```
Processed Data
    ↓ [Run: python train_db_aligned.py]

Random Forest Model trained on:
- Customer purchase frequency
- Recency of last purchase
- Customer age & segment
- Product popularity & price
- Past purchase history

    ↓ [Saved as: db_purchase_prediction_model.pkl]

Collaborative Filtering Model trained on:
- Customer-product interaction matrix
- Matrix factorization (SVD)
- Similar customer patterns
```

### Phase 2: Model Training (DONE ✅)

```
Processed Data
    ↓ [Run: python train_db_aligned.py]

Random Forest Model trained on:
- Customer purchase frequency
- Recency of last purchase
- Customer age & segment
- Product popularity & price
- Past purchase history

    ↓ [Saved as: db_purchase_prediction_model.pkl]

Collaborative Filtering Model trained on:
- Customer-product interaction matrix
- Matrix factorization (SVD)
- Similar customer patterns

    ↓ [Saved as: db_collaborative_filtering_model.pkl]
```

### Phase 3: Model Usage

```
User Input:
- Product: Bread (Product UUID)
- Discount: 10%
- Target: 50 customers

    ↓ [Run: python demo_campaign_generator.py]

AI Processing:
1. Load trained models
2. For each customer, calculate:
   - Purchase probability for Bread
   - Based on their history & behavior
3. Rank all 1000 customers by probability
4. Select top 50

    ↓ [Output: CSV file]

Campaign File:
CustomerID, Name, Location, Probability, ...
CUST00148, Terri Murphy, Gampaha, 100%, ...
CUST00039, Michael Evans, Colombo, 100%, ...
... (50 rows total)

    ↓ [Import to Email System]

Send Emails!
50 targeted emails → 12-13 purchases → High ROI!
```

### Scripts to Run:

| File                             | Purpose                               | When to Use                   |
| -------------------------------- | ------------------------------------- | ----------------------------- |
| `demo_campaign_generator.py`     | Auto demo, generates sample campaigns | First time demo, testing      |
| `interactive_demo.py`            | Menu-driven campaign creator          | Regular use, presentations    |
| `data_analysis/db_preprocessing.py` | Process raw data into features     | When data changes             |
| `train_db_aligned.py`               | Train/retrain the AI models        | Monthly, or when data updates |
| `evaluation/model_evaluation.py` | Check model performance               | After retraining              |

### Files You Get:

| File                                       | Contains                      | Use For                           |
| ------------------------------------------ | ----------------------------- | --------------------------------- |
| `campaign_outputs/campaign_*.csv`          | Customer lists for promotions | Email marketing, SMS campaigns    |
| `campaign_outputs/crosssell_*.csv`         | Cross-sell customer lists     | Upselling, increasing basket size |
| `evaluation/results/evaluation_report.txt` | Performance metrics           | Research paper, presentations     |
| `models/*.pkl`                             | Trained AI models             | System runs automatically         |

## Enhanced Research Components

### Core Features

1. **Purchase Pattern Analysis**: Analyze customer buying history using time-series analysis
2. **Personalized Promotion Targeting**: ML-based customer selection for promotions
3. **Purchase Probability Prediction**: Predict likelihood of customer buying promoted product

### Advanced Features (Research Novelty)

4. **Multi-Armed Bandit Optimization**: Real-time learning of best promotions per customer
5. **Promotion Fatigue Detection**: Identify when customers become less responsive to promotions
6. **Cross-Category Recommendations**: "If customer buys bread, suggest butter" with discounts
7. **Dynamic Discount Optimization**: ML-based optimal discount percentage per customer
8. **Promotion ROI Prediction**: Forecast revenue impact before sending promotions
9. **Customer Lifetime Value Integration**: Prioritize high-value customers
10. **Temporal Pattern Recognition**: Send promotions at optimal times (day/hour)

## Research Contributions

1. **Personalized vs Broadcast**: Compare targeted promotions vs traditional broadcast
2. **Multi-Model Ensemble**: Combine multiple ML approaches for better accuracy
3. **Real-time Adaptation**: Online learning from promotion responses
4. **Explainability**: Why a customer received a specific promotion (XAI)
5. **Fairness Analysis**: Ensure promotions don't discriminate

## Technologies Used

- **Data Generation**: Faker, NumPy, Pandas
- **ML Models**: Scikit-learn, LightGBM, TensorFlow
- **Recommendation**: Surprise, Implicit
- **Analysis**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **API**: FastAPI
- **Deployment**: Docker (optional)

## Evaluation Metrics

- **Precision@K**: Accuracy of top-K customer recommendations
- **Recall@K**: Coverage of interested customers
- **NDCG**: Ranking quality
- **Conversion Rate**: % of customers who purchase after receiving promotion
- **ROI**: Revenue generated vs discount cost
- **Customer Satisfaction**: Promotion relevance score

## 📊 Dataset Specifications

What the system generates:

| Dataset      | Size   | Details                          |
| ------------ | ------ | -------------------------------- |
| Customers    | 1,000  | Age, gender, location, segments  |
| Products     | 250    | 15 categories, realistic pricing |
| Stores       | 10     | Sri Lankan cities                |
| Promotions   | 200    | Various discounts, durations     |
| Transactions | 50,000 | 18 months of realistic purchases |

**Realistic Features:**

- ✅ Customer segments (frequent, regular, occasional, rare)
- ✅ Product affinities (bread → butter, jam)
- ✅ Seasonal patterns (December = more purchases)
- ✅ Promotion responses (price-sensitive vs not)
- ✅ Time-based patterns
