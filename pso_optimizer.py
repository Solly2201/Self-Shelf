import random

def objective_function_profit(test_price, feature_array, cost, model, price_index):
    # Temporarily set the test price in the feature array
    feature_array[price_index] = test_price
    
    # Predict demand using the trained model
    predicted_demand = max(0, model.predict([feature_array])[0])
    
    # Profit = Margin * Volume
    profit = (test_price - cost) * predicted_demand
    
    # TIE-BREAKER LOGIC: If profits are identical across different prices 
    # (due to flat demand predictions), slightly favor the lower price to drive volume.
    volume_incentive = test_price * 0.0001
    
    return profit - volume_incentive

class Particle:
    def __init__(self, bounds):
        self.position = random.uniform(bounds[0], bounds[1])
        self.velocity = random.uniform(-0.5, 0.5)
        self.pbest_position = self.position
        self.pbest_profit = -float('inf')

def run_pso_pricing(features_row, cost, model, bounds, feature_list, num_particles=30, iterations=50):
    price_idx = feature_list.index('PRICE_CURRENT')
    swarm = [Particle(bounds) for _ in range(num_particles)]
    
    gbest_position = random.uniform(bounds[0], bounds[1])
    gbest_profit = -float('inf')
    
    w, c1, c2 = 0.7, 1.5, 1.5 # Inertia and learning factors
    
    for _ in range(iterations):
        for particle in swarm:
            profit = objective_function_profit(particle.position, features_row.copy(), cost, model, price_idx)
            
            # Update personal best
            if profit > particle.pbest_profit:
                particle.pbest_profit = profit
                particle.pbest_position = particle.position
                
            # Update global best
            if profit > gbest_profit:
                gbest_profit = profit
                gbest_position = particle.position
                
        # Move particles
        for particle in swarm:
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (particle.pbest_position - particle.position)
            social = c2 * r2 * (gbest_position - particle.position)
            
            particle.velocity = (w * particle.velocity) + cognitive + social
            particle.position += particle.velocity
            
            # Ensure the particle never breaks your minimum or maximum bounds
            particle.position = max(bounds[0], min(particle.position, bounds[1])) 
            
    return round(gbest_position, 2), round(gbest_profit, 2)