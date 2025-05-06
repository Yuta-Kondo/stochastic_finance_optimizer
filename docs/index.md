# Stochastic Finance Optimizer

A sophisticated Python implementation of stochastic optimization techniques for financial portfolio management, with a focus on both traditional assets and Forex markets.

## Features

- **Stochastic Optimization**: Implementation of stochastic gradient descent for portfolio weight optimization
- **Multi-Asset Support**: Handles both traditional securities and Forex pairs
- **Risk Management**: Sophisticated risk metrics and constraints
- **Performance Analytics**: Comprehensive performance visualization and analysis
- **Modern Portfolio Theory**: Implementation of efficient frontier concepts
- **Dynamic Rebalancing**: Support for periodic portfolio rebalancing

## Quick Start

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

## Mathematical Foundation

The optimization process is based on modern portfolio theory with several enhancements:

1. **Objective Function**:
   \[ \max_w \frac{R_p - R_f}{\sigma_p} \]
   where:
   - \(R_p\) is the portfolio return
   - \(R_f\) is the risk-free rate
   - \(\sigma_p\) is the portfolio volatility

2. **Constraints**:
   \[ \sum_{i=1}^n w_i = 1 \]
   \[ w_i \geq 0 \quad \forall i \]

3. **Risk Calculation**:
   \[ \sigma_p = \sqrt{w^T \Sigma w} \]
   where \(\Sigma\) is the covariance matrix

## Installation

```bash
git clone https://github.com/Yuta-Kondo/stochastic_finance_optimizer.git
cd stochastic_finance_optimizer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Documentation

For detailed documentation, please visit:
- [Installation Guide](installation.md)
- [Usage Guide](usage.md)
- [API Reference](api.md)
- [Technical Report](technical_report.md)
- [Examples](examples/portfolio_optimization.md)
- [Contributing Guide](contributing.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details. 