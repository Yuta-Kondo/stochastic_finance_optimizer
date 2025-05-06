import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.optimizers.stochastic_optimizer import StochasticOptimizer
from src.models.portfolio_model import PortfolioModel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    # Initialize the portfolio model with major currency pairs
    model = PortfolioModel(
        symbols=[
            'EURUSD=X',  # Euro/US Dollar
            'GBPUSD=X',  # British Pound/US Dollar
            'USDJPY=X',  # US Dollar/Japanese Yen
            'AUDUSD=X',  # Australian Dollar/US Dollar
            'USDCAD=X'   # US Dollar/Canadian Dollar
        ],
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Create and run the optimizer
    optimizer = StochasticOptimizer(
        model,
        learning_rate=0.01,
        n_iterations=1000,
        batch_size=20
    )
    
    # Run optimization
    optimal_weights, best_sharpe = optimizer.optimize()
    
    # Calculate final portfolio metrics
    portfolio_return, portfolio_risk = model.calculate_portfolio_metrics(optimal_weights)
    
    # Print results
    print("\nForex Portfolio Optimization Results:")
    print("-" * 40)
    print("Optimal Weights:")
    for symbol, weight in zip(model.symbols, optimal_weights):
        # Clean up symbol names for display
        clean_symbol = symbol.replace('=X', '').replace('USD', '/USD')
        print(f"{clean_symbol}: {weight:.4f}")
    print(f"\nAnnual Portfolio Return: {portfolio_return:.4f}")
    print(f"Annual Portfolio Risk: {portfolio_risk:.4f}")
    print(f"Sharpe Ratio: {best_sharpe:.4f}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    fig = plt.figure(figsize=(15, 10))
    
    # Plot weights
    plt.subplot(2, 2, 1)
    clean_symbols = [s.replace('=X', '').replace('USD', '/USD') for s in model.symbols]
    plt.bar(clean_symbols, optimal_weights)
    plt.title('Optimal Portfolio Weights')
    plt.xticks(rotation=45)
    plt.ylabel('Weight')
    
    # Plot cumulative returns
    plt.subplot(2, 2, 2)
    cumulative_returns = (1 + model.returns).cumprod()
    for symbol in model.symbols:
        clean_symbol = symbol.replace('=X', '').replace('USD', '/USD')
        plt.plot(cumulative_returns.index, cumulative_returns[symbol], label=clean_symbol)
    plt.title('Cumulative Returns')
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Cumulative Return')
    
    # Plot correlation heatmap
    plt.subplot(2, 2, 3)
    correlation_matrix = model.returns.corr()
    clean_symbols = [s.replace('=X', '').replace('USD', '/USD') for s in correlation_matrix.index]
    sns.heatmap(correlation_matrix, 
                xticklabels=clean_symbols,
                yticklabels=clean_symbols,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f')
    plt.title('Currency Pair Correlations')
    
    # Plot rolling volatility
    plt.subplot(2, 2, 4)
    rolling_vol = model.returns.rolling(window=30).std() * np.sqrt(252)
    for symbol in model.symbols:
        clean_symbol = symbol.replace('=X', '').replace('USD', '/USD')
        plt.plot(rolling_vol.index, rolling_vol[symbol], label=clean_symbol)
    plt.title('30-Day Rolling Volatility')
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Annualized Volatility')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 