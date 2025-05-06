import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple

class PortfolioModel:
    """
    A class implementing portfolio optimization models for Forex trading.
    """
    
    def __init__(self, symbols=None, start_date='2020-01-01', end_date='2023-12-31'):
        # Default to major currency pairs
        self.symbols = symbols or [
            'EURUSD=X',  # Euro/US Dollar
            'GBPUSD=X',  # British Pound/US Dollar
            'USDJPY=X',  # US Dollar/Japanese Yen
            'AUDUSD=X',  # Australian Dollar/US Dollar
            'USDCAD=X'   # US Dollar/Canadian Dollar
        ]
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.load_data()
        
    def load_data(self):
        """Load Forex data and calculate returns"""
        # Download all currency pairs at once
        data = yf.download(self.symbols, start=self.start_date, end=self.end_date)['Close']
        
        # Handle single currency pair case
        if len(self.symbols) == 1:
            data = pd.DataFrame(data)
            data.columns = self.symbols
        
        self.data = data
        # Calculate daily returns
        self.returns = self.data.pct_change().dropna()
        
    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio return and risk"""
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Calculate portfolio metrics
        # For Forex, we use 252 trading days for annualization
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_risk = np.sqrt(weights.T @ self.returns.cov() @ weights) * np.sqrt(252)
        
        return portfolio_return, portfolio_risk
    
    def get_data(self):
        """Return the loaded data"""
        return self.data, self.returns
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio return"""
        return np.sum(self.returns.mean() * weights) * 252
    
    def portfolio_risk(self, weights: np.ndarray) -> float:
        """Calculate portfolio risk (standard deviation)"""
        return np.sqrt(weights.T @ self.returns.cov() @ weights) * np.sqrt(252)
    
    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        portfolio_return = self.portfolio_return(weights)
        portfolio_risk = self.portfolio_risk(weights)
        # Using a lower risk-free rate for Forex
        risk_free_rate = 0.01  # 1% risk-free rate
        return (portfolio_return - risk_free_rate) / portfolio_risk
    
    def get_constraints(self) -> List[Dict]:
        """Get optimization constraints"""
        return [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds"""
        return [(0, 1) for _ in range(len(self.symbols))]  # weights between 0 and 1
    
    def optimize_portfolio(
        self,
        optimizer: 'StochasticOptimizer',
        initial_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Optimize portfolio weights.
        
        Args:
            optimizer: StochasticOptimizer instance
            initial_weights: Initial weight values
            
        Returns:
            Dictionary containing optimization results
        """
        if initial_weights is None:
            initial_weights = np.array([1/len(self.symbols)] * len(self.symbols))
            
        result = optimizer.optimize(
            initial_point=initial_weights,
            method='stochastic_gradient_descent'
        )
        
        return {
            'weights': result['optimal_point'],
            'expected_return': self.portfolio_return(result['optimal_point']),
            'risk': self.portfolio_risk(result['optimal_point']),
            'sharpe_ratio': -result['objective_value'],
            'convergence_history': result['history']
        } 