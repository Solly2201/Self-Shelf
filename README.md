# Self-Shelf

### AI-Powered Dynamic Pricing for Retail Inventory

Self-Shelf is a machine learning-based retail pricing optimization system that combines **demand prediction**, **expiry-aware inventory logic**, and **Particle Swarm Optimization (PSO)** to recommend product prices.

The system is designed around a simple retail objective:

> **Sell products at the highest practical price while accounting for demand and inventory urgency.**

For fresh inventory, the system focuses on expected profit. For products approaching expiry, it forces markdowns to encourage clearance and reduce potential waste.

---

## How It Works

Self-Shelf follows a two-stage optimization pipeline:

```text
Retail Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ├── Cost
     ├── Shelf Life
     ├── Days to Expiry
     └── Urgency Ratio
     │
     ▼
Random Forest Regressor
     │
     ▼
Demand Prediction
     │
     ▼
Fresh / Urgent Classification
     │
     ▼
Particle Swarm Optimization
     │
     ▼
AI-Optimized Price
     │
     ▼
Pricing Action + CSV Output
```

The implementation separates **demand modeling** from **price optimization**. The trained Random Forest acts as the demand simulator, while PSO searches the permitted price range for a profitable price.

---

## Key Features

### 1. Demand Prediction

A **Random Forest Regressor** predicts expected demand using:

- Current price
- Retail price
- Estimated cost
- Promotion status
- Maximum shelf life
- Urgency ratio
- Days to expiry

The current implementation trains a Random Forest with:

- **100 estimators**
- **Maximum depth: 10**
- **80/20 train-test split**

---

### 2. Profit-Based Price Optimization

For each product, candidate prices are evaluated using:

```text
Expected Profit = (Price - Cost) × Predicted Demand
```

The PSO optimizer searches through possible prices rather than testing every price manually.

A small volume incentive is also included when candidate prices produce effectively identical profit values, slightly favoring the lower price.

---

### 3. Expiry-Aware Pricing

Self-Shelf does not treat every product identically.

Products are classified using their expiry-related features.

#### Fresh Inventory

Fresh products use a pricing range designed to protect the existing margin while allowing markdowns when beneficial.

Possible actions:

```text
Price Maintained
Markdown Applied
```

#### Urgent Inventory

A product becomes urgent when:

```text
Days to Expiry ≤ 3
```

or its urgency ratio crosses the configured threshold.

Urgent products are forced into a markdown range:

```text
Current Price × 30%
        ↓
Current Price × 90%
```

This prevents the optimizer from maintaining or increasing the price when the primary objective should be clearing inventory.

Action:

```text
Markdown Applied (Clearance)
```

---

## Shelf-Life Model

The current implementation assigns department-level shelf-life assumptions:

| Department | Maximum Shelf Life |
|---|---:|
| Bakery | 7 days |
| Deli | 5 days |
| Snacks | 90 days |
| Beverages | 120 days |
| Frozen Foods | 180 days |
| Other | 30 days |

Days to expiry and the urgency ratio are then derived from these values.

---

## Project Structure

```text
Self-Shelf/
│
├── main.py
│   └── Main data pipeline, feature engineering,
│       model training, inventory logic and output
│
├── pso_optimizer.py
│   └── Particle Swarm Optimization implementation
│
├── walmart_large_sample_data_with_categories.csv
│   └── Retail dataset used by the pipeline
│
└── final_optimized_prices.csv
    └── Generated optimized pricing results
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical operations and feature engineering |
| Scikit-learn | Machine learning |
| Random Forest | Demand prediction |
| Particle Swarm Optimization | Price optimization |
| CSV | Dataset and result storage |

---

## Dataset

The project uses a Walmart retail dataset containing product, department, pricing, promotion, and related retail attributes.

The pipeline samples **5,000 records** from the available dataset for processing.

During preprocessing, additional variables are derived, including:

```text
COST
MAX_SHELF_LIFE
DAYS_TO_EXPIRY
URGENCY_RATIO
DEMAND
```

The current prototype estimates product cost as:

```text
COST = PRICE_RETAIL × 0.70
```

Demand is then generated from the engineered pricing, promotion, and expiry-related features for the prototype's modeling pipeline.

---

## Processing Pipeline

### Step 1 — Load Data

The application loads:

```text
walmart_large_sample_data_with_categories.csv
```

and samples 5,000 records.

### Step 2 — Clean Data

The pipeline:

- Removes records missing required price/product fields
- Removes non-positive prices
- Cleans department and product-name strings
- Converts promotion information into a binary indicator

### Step 3 — Engineer Features

The system creates cost, shelf-life, expiry, urgency, and demand features.

### Step 4 — Train Demand Model

The data is split into training and testing sets and used to train the Random Forest demand model.

### Step 5 — Determine Inventory Status

Each selected product is classified as either:

```text
Fresh
```

or:

```text
Urgent (Clearance)
```

### Step 6 — Optimize Price

PSO searches the permitted price range while repeatedly querying the Random Forest for predicted demand.

Each particle represents a candidate price.

The swarm updates:

- Particle position
- Particle velocity
- Personal best
- Global best

until the optimization converges toward a high-profit candidate.

### Step 7 — Generate Pricing Output

Results are written to:

```text
final_optimized_prices.csv
```

---

## Particle Swarm Optimization

The PSO implementation uses the standard swarm concepts:

```text
Particle Position
       │
       ▼
Candidate Price
       │
       ▼
Predicted Demand
       │
       ▼
Expected Profit
       │
       ▼
Personal Best / Global Best
       │
       ▼
Updated Velocity
       │
       ▼
New Candidate Price
```

The current pipeline calls the optimizer with:

- **20 particles**
- **30 iterations**
- **Inertia weight:** `0.7`
- **Cognitive coefficient:** `1.5`
- **Social coefficient:** `1.5`

Price positions are always constrained to the configured lower and upper bounds.

---

## Running the Project

### Prerequisites

Install:

- Python 3.x
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Solly2201/Self-Shelf.git
cd Self-Shelf
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

### 3. Run Self-Shelf

```bash
python main.py
```

You will be prompted:

```text
Enter number of values you want to optimize:
```

For example:

```text
Enter number of values you want to optimize: 10
```

The system will then train the model, optimize the requested products, and generate the output CSV.

---

## Output

The generated `final_optimized_prices.csv` contains:

| Field | Description |
|---|---|
| `SKU` | Product identifier |
| `Product_Name` | Product name |
| `Department` | Product department |
| `Wholesale_Cost` | Estimated product cost |
| `Old_Walmart_Price` | Existing price |
| `New_AI_Optimized_Price` | Recommended price |
| `Days_To_Expiry` | Estimated remaining shelf life |
| `Urgency_Status` | Fresh / Urgent |
| `Action_Taken` | Pricing action |

Example:

```text
SKU        Old Price    AI Price    Days    Status
SKU001     $4.99        $4.49       2       Urgent
SKU002     $7.99        $7.99       30      Fresh
```

---

## Design Decisions

### Why Random Forest?

Random Forest provides a practical nonlinear regression model for capturing relationships between pricing, promotions, inventory characteristics, and demand without requiring a strictly linear relationship.

### Why PSO?

The optimization problem is continuous: the system needs to search through a range of possible prices.

PSO is suitable for this because it can explore a continuous search space using a population of candidate solutions while balancing exploration and exploitation.

### Why Separate Prediction and Optimization?

The model answers:

```text
"What demand should we expect at this price?"
```

The optimizer answers:

```text
"Given that demand, what price should we choose?"
```

Separating these responsibilities makes the pricing pipeline easier to reason about and extend.

---

## Limitations

Self-Shelf is a prototype and the current implementation has several limitations:

- Demand behavior is dependent on the available dataset and engineered assumptions.
- Product cost is estimated rather than obtained from real procurement data.
- Days to expiry are simulated during preprocessing.
- Demand is generated using prototype business logic rather than learned from actual historical sales volume.
- Random Forest is not designed to reliably extrapolate far outside the observed training distribution.
- The system currently operates on batch data rather than a live POS stream.
- Pricing is optimized for selected products rather than a continuously running store-wide system.
- External factors such as competitors, weather, local events, and real-time customer behavior are not currently modeled.

These limitations make the current project best suited as an academic/prototype demonstration of an AI-assisted retail pricing pipeline.

---

## Future Scope

Potential extensions include:

- Real-time POS transaction streaming
- Kafka-based event processing
- Live inventory integration
- Real historical demand modeling
- Deep learning-based demand forecasting
- Reinforcement learning for dynamic pricing
- Competitor price monitoring
- Customer segmentation
- Personalized promotions
- Multi-store optimization
- Distributed optimization using Spark
- Explainable AI for pricing recommendations
- Integration with Electronic Shelf Labels (ESLs)
- ERP integration for automated price updates

---

## Repository

**GitHub:**  
https://github.com/Solly2201/Self-Shelf

---

## Author

**Shreshtha Bindal**

Computer Engineering  
Mukesh Patel School of Technology Management & Engineering (MPSTME), NMIMS, Mumbai

GitHub: https://github.com/Solly2201

---

## License

This project is intended for academic and educational purposes.
