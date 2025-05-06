# Usage Guide

## Basic Usage

### Portfolio Model

The `PortfolioModel` class is the core component for managing financial data and calculations:

```python
from src.models import PortfolioModel

# Initialize with default currency pairs
model = PortfolioModel()

# Or specify custom symbols
model = PortfolioModel(
    symbols=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Load and prepare data
model.load_data()

# Get portfolio metrics
returns, risk = model.calculate_portfolio_metrics(weights)
```

### Stochastic Optimizer

The `StochasticOptimizer` class implements the optimization algorithm:

```python
from src.optimizers import StochasticOptimizer

# Create optimizer instance
optimizer = StochasticOptimizer(
    model,
    learning_rate=0.01,
    n_iterations=1000,
    batch_size=20
)

# Run optimization
optimal_weights, sharpe_ratio = optimizer.optimize()
```

## Advanced Usage

### Custom Constraints

You can add custom constraints to the optimization:

```python
def custom_constraint(weights):
    # Example: Maximum position size of 30%
    return np.max(weights) <= 0.3

optimizer.add_constraint(custom_constraint)
```

### Risk Management

The model includes various risk management features:

```python
# Calculate Value at Risk (VaR)
var_95 = model.calculate_var(weights, confidence=0.95)

# Calculate Maximum Drawdown
max_drawdown = model.calculate_max_drawdown(weights)

# Calculate Rolling Volatility
rolling_vol = model.calculate_rolling_volatility(window=30)
```

### Performance Analysis

Analyze portfolio performance with various metrics:

```python
# Calculate Sharpe Ratio
sharpe = model.sharpe_ratio(weights)

# Calculate Sortino Ratio
sortino = model.sortino_ratio(weights)

# Calculate Information Ratio
info_ratio = model.information_ratio(weights, benchmark_returns)
```

## Visualization

The package includes comprehensive visualization tools:

```python
# Plot portfolio weights
model.plot_weights(weights)

# Plot cumulative returns
model.plot_cumulative_returns()

# Plot correlation heatmap
model.plot_correlation_matrix()

# Plot rolling volatility
model.plot_rolling_volatility(window=30)
```

## Best Practices

1. **Data Quality**
   - Always check for missing data
   - Handle outliers appropriately
   - Use appropriate time periods for analysis

2. **Optimization**
   - Start with reasonable initial weights
   - Monitor convergence
   - Validate results with different parameters

3. **Risk Management**
   - Set appropriate position limits
   - Monitor correlation changes
   - Implement stop-loss mechanisms

4. **Performance Monitoring**
   - Track key metrics regularly
   - Monitor transaction costs
   - Rebalance when necessary 