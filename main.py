import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
from pso_optimizer import run_pso_pricing 

# Mute the scikit-learn feature names warning
warnings.filterwarnings("ignore", message="x does not have valid feature names")

def main():
    try:
        n = int(input("Enter number of values you want to optimize: "))
    except ValueError:
        print("Please enter a valid whole number.")
        return
    
    if n <= 0:
        print("Number of items must be greater than 0.")
        return
        
    print("1. Loading Data From The CSV File...")
    try:
        df = pd.read_csv('walmart_large_sample_data_with_categories.csv')
    except FileNotFoundError:
        print("Could not find 'walmart_large_sample_data_with_categories.csv'.")
        return
        
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    
    print("2. Cleaning Data...")
    df = df.dropna(subset=['PRICE_RETAIL', 'PRICE_CURRENT', 'DEPARTMENT', 'PRODUCT_NAME'])
    df = df[(df['PRICE_RETAIL'] > 0) & (df['PRICE_CURRENT'] > 0)]
    
    df['DEPARTMENT'] = df['DEPARTMENT'].str.strip()
    df['PRODUCT_NAME'] = df['PRODUCT_NAME'].str.strip()
    
    df['PROMOTION'] = df['PROMOTION'].fillna(0)
    df['PROMOTION'] = np.where(df['PROMOTION'] == 0, 0, 1)
    
    print("3. Engineering Advanced Expiry & Business Features...")
    df['COST'] = df['PRICE_RETAIL'] * 0.70
    
    shelf_life_map = {
        'Bakery': 7,
        'Deli': 5,
        'Snacks': 90,
        'Beverages': 120,
        'Frozen Foods': 180
    }
    df['MAX_SHELF_LIFE'] = df['DEPARTMENT'].map(shelf_life_map).fillna(30)
    
    np.random.seed(42)
    df['DAYS_TO_EXPIRY'] = np.ceil(np.random.uniform(1, df['MAX_SHELF_LIFE'])).astype(int)
    df['URGENCY_RATIO'] = df['DAYS_TO_EXPIRY'] / df['MAX_SHELF_LIFE']
    
    random_discounts = np.random.uniform(0.10, 0.40, size=len(df)) 
    is_historically_discounted = np.random.choice([True, False], p=[0.3, 0.7], size=len(df))
    
    df['PRICE_CURRENT'] = np.where(is_historically_discounted, 
                                df['PRICE_RETAIL'] * (1 - random_discounts), 
                                df['PRICE_CURRENT'])
    
    discount_depth = df['PRICE_RETAIL'] - df['PRICE_CURRENT']
    base_demand = 20 + (df['PROMOTION'] * 15)
    
    is_urgent = (df['URGENCY_RATIO'] <= 0.20) | (df['DAYS_TO_EXPIRY'] <= 3)
    
    urgent_demand = (base_demand * 0.2) + (discount_depth * 40) 
    normal_demand = base_demand + (discount_depth * 10)        
    
    df['DEMAND'] = np.where(is_urgent, urgent_demand, normal_demand).clip(min=1).astype(int)

    print("4. Training the AI Model...")
    features = ['PRICE_CURRENT', 'PRICE_RETAIL', 'COST', 'PROMOTION', 'MAX_SHELF_LIFE', 'URGENCY_RATIO','DAYS_TO_EXPIRY']
    x = df[features]
    y = df['DEMAND']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    demand_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    demand_model.fit(x_train.values, y_train.values)
    
    print("AI Trained Successfully!")

    print(f"5. Running Optimization on {n} Items...")
    results = []
    
    max_items = min(n, len(x_test))
    test_indices = x_test.index[:max_items] 
    
    for i, idx in enumerate(test_indices):
        sku = df.loc[idx].get('SKU', f'UNKNOWN-{idx}')
        product_name = df.loc[idx, 'PRODUCT_NAME']
        department = df.loc[idx, 'DEPARTMENT']
        days_to_expiry = df.loc[idx, 'DAYS_TO_EXPIRY']
        urgency_ratio = df.loc[idx, 'URGENCY_RATIO']
        
        test_scenario = x_test.loc[idx].values
        current_price = x_test.loc[idx, 'PRICE_CURRENT']
        cost = x_test.loc[idx, 'COST']
        retail_price = x_test.loc[idx, 'PRICE_RETAIL']
        is_item_urgent = (urgency_ratio <= 0.20) or (days_to_expiry <= 3)

        if is_item_urgent:
            # Drop the 'cost' safety net entirely. Force upper bound down.
            lower_bound = current_price * 0.30 # Allow up to 70% off
            # FORCE a markdown by restricting the highest price the AI can pick
            upper_bound = current_price * 0.90 
            
            # Sunk Cost: Tell the AI cost is $0 so it maximizes Revenue, not Margin
            pso_cost = 0.0 
        else:
            # FRESH: Protect the margin, but ensure lower bound never forces a price increase.
            lower_bound = min(cost * 1.05, current_price * 0.95)
            upper_bound = current_price 
            pso_cost = cost
            
        # highest price can NEVER exceed current price or Retail (MRP)
        upper_bound = min(upper_bound, current_price, retail_price)
        
        # Prevent boundary crashes
        if lower_bound > upper_bound:
            lower_bound = upper_bound * 0.90
            
        price_bounds = (lower_bound, upper_bound)

        optimal_price, projected_profit = run_pso_pricing(
            features_row=test_scenario, 
            cost=pso_cost, # Use the dynamic cost metric
            model=demand_model, 
            bounds=price_bounds, 
            feature_list=features, 
            num_particles=20, 
            iterations=30
        )
        
        # Hard limit the final output just in case of float precision issues
        optimal_price = min(optimal_price, current_price, retail_price)

        if is_item_urgent:
            # If it's urgent, it MUST be a markdown. Force it if floating math was slightly off.
            if optimal_price >= current_price:
                optimal_price = current_price * 0.90
            action_taken = "Markdown Applied (Clearance)"
        else:
            if abs(optimal_price - current_price) <= 0.05:
                optimal_price = current_price
                action_taken = "Price Maintained"
            elif optimal_price < current_price:
                action_taken = "Markdown Applied"
            else:
                optimal_price = current_price
                action_taken = "Price Maintained"

        results.append({
            'SKU': sku,
            'Product_Name': product_name,
            'Department': department,
            'Wholesale_Cost': round(cost, 2),
            'Old_Walmart_Price': round(current_price, 2),
            'New_AI_Optimized_Price': round(optimal_price, 2),
            'Days_To_Expiry': days_to_expiry,
            'Urgency_Status': "Urgent (Clearance)" if is_item_urgent else "Fresh",
            'Action_Taken': action_taken
        })
        
        if (i+1) % 10 == 0:
            print(f"Optimized {i+1}/{max_items} items...")
            
    results_df = pd.DataFrame(results)
    
    try:
        results_df.to_csv('final_optimized_prices.csv', index=False)
        print("\nThe Prices have been optimized successfully and the results have been stored in 'final_optimized_prices.csv'.")
    except PermissionError:
        print("\nPermission Error: Could not save the file!")
        print("Please close 'final_optimized_prices.csv' in Excel and press Enter to try saving again.")
        input("Press Enter when you have closed the file...")
        results_df.to_csv('final_optimized_prices.csv', index=False)
        print("\nThe Prices have been optimized successfully and the results have been stored in 'final_optimized_prices.csv'.")

if __name__ == "__main__":
    main()