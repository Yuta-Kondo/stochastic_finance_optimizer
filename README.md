# Stochastic Finance Optimizer

A sophisticated Python implementation of stochastic optimization techniques for financial portfolio management, with a focus on both traditional assets and Forex markets. This project demonstrates advanced portfolio optimization techniques using stochastic gradient descent and modern risk management approaches.

## Key Features

- **Stochastic Optimization**: Implementation of stochastic gradient descent for portfolio weight optimization
- **Multi-Asset Support**: Handles both traditional securities and Forex pairs
- **Risk Management**: Sophisticated risk metrics and constraints
- **Performance Analytics**: Comprehensive performance visualization and analysis
- **Modern Portfolio Theory**: Implementation of efficient frontier concepts
- **Dynamic Rebalancing**: Support for periodic portfolio rebalancing

## Technical Implementation

### Optimization Algorithm

The core optimization algorithm uses stochastic gradient descent (SGD) with the following key features:

- **Objective Function**: Maximizes the Sharpe Ratio while maintaining portfolio constraints
- **Batch Processing**: Mini-batch sampling for efficient gradient estimation
- **Adaptive Learning**: Learning rate optimization for convergence
- **Constraint Handling**: Efficient projection onto the simplex for weight constraints

### Risk Management

The system implements several risk management features:

- **Volatility Analysis**: Rolling and exponential volatility estimation
- **Correlation Management**: Dynamic correlation analysis for diversification
- **Value at Risk (VaR)**: Historical VaR calculation
- **Maximum Drawdown**: Tracking of maximum portfolio drawdown

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stochastic_finance_optimizer.git
cd stochastic_finance_optimizer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Example

```python
from src.models import PortfolioModel
from src.optimizers import StochasticOptimizer

# Initialize with Forex pairs
model = PortfolioModel(
    symbols=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Create and run optimizer
optimizer = StochasticOptimizer(
    model,
    learning_rate=0.01,
    n_iterations=1000,
    batch_size=20
)

# Get optimal weights and performance metrics
optimal_weights, sharpe_ratio = optimizer.optimize()
returns, risk = model.calculate_portfolio_metrics(optimal_weights)
```

## Results Visualization

The package includes comprehensive visualization tools:

- Portfolio weight distribution
- Cumulative returns analysis
- Correlation heatmaps
- Rolling volatility charts
- Performance attribution analysis

Running the example script will generate four plots:
1. Optimal portfolio weights for each currency pair
2. Cumulative returns over time
3. Correlation matrix between currency pairs
4. 30-day rolling volatility for each pair

## Mathematical Foundation

The optimization process is based on modern portfolio theory with several enhancements:

1. **Objective Function**:
   $$ \max_w \frac{R_p - R_f}{\sigma_p} $$
   where:
   - $R_p$ is the portfolio return
   - $R_f$ is the risk-free rate
   - $\sigma_p$ is the portfolio volatility

2. **Constraints**:
   $$ \sum_{i=1}^n w_i = 1 $$
   $$ w_i \geq 0 \quad \forall i $$

3. **Risk Calculation**:
   $$ \sigma_p = \sqrt{w^T \Sigma w} $$
   where $\Sigma$ is the covariance matrix

4. **Gradient of Sharpe Ratio**:
   $$ \nabla_w SR = \frac{1}{\sigma_p} \left(\nabla_w R_p - SR \cdot \nabla_w \sigma_p\right) $$

5. **Portfolio Return**:
   $$ R_p = \sum_{i=1}^n w_i R_i $$
   where $R_i$ is the return of asset $i$

6. **Portfolio Variance**:
   $$ \sigma_p^2 = w^T \Sigma w = \sum_{i=1}^n \sum_{j=1}^n w_i w_j \sigma_{ij} $$
   where $\sigma_{ij}$ is the covariance between assets $i$ and $j$

## Project Structure

```
stochastic_finance_optimizer/
├── src/
│   ├── optimizers/        # Optimization algorithms
│   ├── models/           # Financial models
│   ├── risk/            # Risk management tools
│   └── utils/           # Utility functions
├── tests/               # Unit tests
├── docs/               # Documentation
│   └── images/        # Result visualizations
├── notebooks/         # Jupyter notebooks with examples
└── examples/         # Example scripts
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 