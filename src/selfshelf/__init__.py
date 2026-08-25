"""Self-Shelf: expiry-aware dynamic pricing engine.

Architecture:

    Demand Model (ML)   -> baseline demand at the current price
    Economic Layer      -> elasticity, expiry pressure, inventory pressure,
                           expected waste, constraints
    PSO                 -> searches the constrained price range for the best
                           economic score
    Recommendation      -> price, action, and structured explanation
"""

__version__ = "0.2.0"
