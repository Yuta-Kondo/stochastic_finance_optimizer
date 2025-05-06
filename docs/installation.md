# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Yuta-Kondo/stochastic_finance_optimizer.git
cd stochastic_finance_optimizer
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Verifying Installation

To verify the installation, run the example script:

```bash
python src/examples/portfolio_optimization.py
```

You should see output showing the optimization results and plots.

## Troubleshooting

### Common Issues

1. **Package Installation Errors**
   - Make sure you have the latest pip version:
     ```bash
     pip install --upgrade pip
     ```
   - Try installing packages one by one if batch installation fails

2. **Virtual Environment Issues**
   - If you get permission errors, try:
     ```bash
     python -m venv venv --clear
     ```
   - Make sure you're using the correct Python version

3. **Data Download Issues**
   - Check your internet connection
   - Verify that yfinance is properly installed
   - Try updating yfinance:
     ```bash
     pip install --upgrade yfinance
     ```

## Development Installation

For development, install additional dependencies:

```bash
pip install -r requirements-dev.txt
```

This includes:
- pytest for testing
- black for code formatting
- flake8 for linting
- mypy for type checking 