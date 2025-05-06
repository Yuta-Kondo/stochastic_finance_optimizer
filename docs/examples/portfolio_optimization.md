# Portfolio Optimization Example

This example demonstrates how to use the Stochastic Finance Optimizer to optimize a portfolio of currency pairs.

## Basic Example

```python
from src.models import PortfolioModel
from src.optimizers import StochasticOptimizer

# Initialize the portfolio model with major currency pairs
model = PortfolioModel(
    symbols=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Create the optimizer
optimizer = StochasticOptimizer(
    model,
    learning_rate=0.01,
    n_iterations=1000,
    batch_size=20
)

# Run optimization
optimal_weights, sharpe_ratio = optimizer.optimize()

# Print results
print("Optimal Weights:")
for symbol, weight in zip(model.symbols, optimal_weights):
    print(f"{symbol}: {weight:.4f}")
print(f"\nSharpe Ratio: {sharpe_ratio:.4f}")
```

## Advanced Example with Custom Constraints

```python
import numpy as np

def max_position_constraint(weights):
    return np.max(weights) <= 0.3

def sector_exposure_constraint(weights):
    # Example: Limit exposure to USD pairs
    usd_pairs = [0, 1, 2]  # Indices of USD pairs
    return np.sum(weights[usd_pairs]) <= 0.6

# Add constraints to optimizer
optimizer.add_constraint(max_position_constraint)
optimizer.add_constraint(sector_exposure_constraint)

# Run optimization with constraints
optimal_weights, sharpe_ratio = optimizer.optimize()
```

## Visualization

The package includes various visualization tools:

```python
# Plot portfolio weights
model.plot_weights(optimal_weights)

# Plot cumulative returns
model.plot_cumulative_returns()

# Plot correlation matrix
model.plot_correlation_matrix()

# Plot rolling volatility
model.plot_rolling_volatility(window=30)
```

## Expected Output

The example will produce:
1. Optimal portfolio weights for each currency pair
2. Portfolio metrics (return, risk, Sharpe ratio)
3. Various visualizations of the portfolio characteristics

## Notes

- The example uses major currency pairs, but you can use any available currency pairs
- The optimization parameters can be adjusted based on your needs
- Consider transaction costs and liquidity when implementing in practice 