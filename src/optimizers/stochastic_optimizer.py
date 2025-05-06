import numpy as np
from typing import Tuple, Optional

class StochasticOptimizer:
    def __init__(self, model, learning_rate=0.01, n_iterations=1000, batch_size=20):
        self.model = model
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
    
    def optimize(self, initial_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Optimize portfolio weights using stochastic gradient descent
        """
        if initial_weights is None:
            n_assets = len(self.model.symbols)
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()
        else:
            weights = initial_weights.copy()
        
        best_sharpe = float('-inf')
        best_weights = weights.copy()
        
        for _ in range(self.n_iterations):
            # Get random batch of returns
            returns = self.model.returns.sample(n=min(self.batch_size, len(self.model.returns)))
            
            # Calculate gradients
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_risk = np.sqrt(weights.T @ returns.cov() @ weights) * np.sqrt(252)
            
            # Calculate Sharpe ratio gradient
            risk_free_rate = 0.02  # Assuming 2% risk-free rate
            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights.copy()
            
            # Update weights using gradient ascent
            gradient = (returns.mean() * 252) / portfolio_risk - \
                      (portfolio_return - risk_free_rate) * \
                      (returns.cov() @ weights) * np.sqrt(252) / (portfolio_risk ** 2)
            
            weights = weights + self.learning_rate * gradient
            
            # Project weights back to simplex (ensure they sum to 1 and are non-negative)
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
        
        return best_weights, best_sharpe 