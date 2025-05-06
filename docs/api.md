# API Documentation

## PortfolioModel

The `PortfolioModel` class handles data management and portfolio calculations.

### Initialization

```python
PortfolioModel(
    symbols: List[str] = None,
    start_date: str = '2020-01-01',
    end_date: str = '2023-12-31'
)
```

**Parameters:**
- `symbols`: List of currency pair symbols (default: major Forex pairs)
- `start_date`: Start date for data (format: 'YYYY-MM-DD')
- `end_date`: End date for data (format: 'YYYY-MM-DD')

### Methods

#### load_data()
Loads and prepares historical data for the specified currency pairs.

**Returns:**
- None

#### calculate_portfolio_metrics(weights: np.ndarray)
Calculates portfolio return and risk.

**Parameters:**
- `weights`: Array of portfolio weights

**Returns:**
- `tuple`: (portfolio_return, portfolio_risk)

#### calculate_var(weights: np.ndarray, confidence: float = 0.95)
Calculates Value at Risk.

**Parameters:**
- `weights`: Array of portfolio weights
- `confidence`: Confidence level (default: 0.95)

**Returns:**
- `float`: Value at Risk

#### calculate_max_drawdown(weights: np.ndarray)
Calculates maximum drawdown.

**Parameters:**
- `weights`: Array of portfolio weights

**Returns:**
- `float`: Maximum drawdown

#### calculate_rolling_volatility(window: int = 30)
Calculates rolling volatility.

**Parameters:**
- `window`: Rolling window size (default: 30)

**Returns:**
- `pd.Series`: Rolling volatility

## StochasticOptimizer

The `StochasticOptimizer` class implements the optimization algorithm.

### Initialization

```python
StochasticOptimizer(
    model: PortfolioModel,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    batch_size: int = 20
)
```

**Parameters:**
- `model`: PortfolioModel instance
- `learning_rate`: Learning rate for gradient descent
- `n_iterations`: Number of optimization iterations
- `batch_size`: Size of mini-batches for stochastic gradient descent

### Methods

#### optimize(initial_weights: np.ndarray = None)
Runs the optimization process.

**Parameters:**
- `initial_weights`: Initial portfolio weights (optional)

**Returns:**
- `tuple`: (optimal_weights, best_sharpe_ratio)

#### add_constraint(constraint_func: Callable)
Adds a custom constraint to the optimization.

**Parameters:**
- `constraint_func`: Function that takes weights and returns a boolean

**Returns:**
- None

## Visualization Methods

### plot_weights(weights: np.ndarray)
Plots portfolio weights.

**Parameters:**
- `weights`: Array of portfolio weights

**Returns:**
- `matplotlib.figure.Figure`: Plot figure

### plot_cumulative_returns()
Plots cumulative returns for all assets.

**Returns:**
- `matplotlib.figure.Figure`: Plot figure

### plot_correlation_matrix()
Plots correlation matrix heatmap.

**Returns:**
- `matplotlib.figure.Figure`: Plot figure

### plot_rolling_volatility(window: int = 30)
Plots rolling volatility.

**Parameters:**
- `window`: Rolling window size (default: 30)

**Returns:**
- `matplotlib.figure.Figure`: Plot figure 