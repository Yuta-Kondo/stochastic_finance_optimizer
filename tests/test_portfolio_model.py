import pytest
import numpy as np
from src.models.portfolio_model import PortfolioModel
from src.optimizers.stochastic_optimizer import StochasticOptimizer

@pytest.fixture
def sample_portfolio():
    tickers = ['AAPL', 'MSFT']
    return PortfolioModel(
        tickers=tickers,
        start_date='2023-01-01',
        end_date='2023-12-31',
        risk_free_rate=0.02
    )

def test_portfolio_initialization(sample_portfolio):
    assert len(sample_portfolio.tickers) == 2
    assert sample_portfolio.returns is not None
    assert sample_portfolio.covariance is not None
    assert sample_portfolio.mean_returns is not None

def test_portfolio_return(sample_portfolio):
    weights = np.array([0.5, 0.5])
    return_value = sample_portfolio.portfolio_return(weights)
    assert isinstance(return_value, float)
    assert not np.isnan(return_value)

def test_portfolio_risk(sample_portfolio):
    weights = np.array([0.5, 0.5])
    risk_value = sample_portfolio.portfolio_risk(weights)
    assert isinstance(risk_value, float)
    assert risk_value >= 0
    assert not np.isnan(risk_value)

def test_sharpe_ratio(sample_portfolio):
    weights = np.array([0.5, 0.5])
    sharpe = sample_portfolio.sharpe_ratio(weights)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)

def test_constraints(sample_portfolio):
    constraints = sample_portfolio.get_constraints()
    assert len(constraints) > 0
    assert all('type' in c for c in constraints)
    assert all('fun' in c for c in constraints)

def test_bounds(sample_portfolio):
    bounds = sample_portfolio.get_bounds()
    assert len(bounds) == len(sample_portfolio.tickers)
    assert all(len(b) == 2 for b in bounds)
    assert all(0 <= b[0] <= b[1] <= 1 for b in bounds)

def test_optimization(sample_portfolio):
    optimizer = StochasticOptimizer(
        objective_function=sample_portfolio.objective_function,
        constraints=sample_portfolio.get_constraints(),
        bounds=sample_portfolio.get_bounds()
    )
    
    result = sample_portfolio.optimize_portfolio(optimizer)
    
    assert 'weights' in result
    assert 'expected_return' in result
    assert 'risk' in result
    assert 'sharpe_ratio' in result
    assert 'convergence_history' in result
    
    assert len(result['weights']) == len(sample_portfolio.tickers)
    assert np.isclose(np.sum(result['weights']), 1.0, atol=1e-6)
    assert all(0 <= w <= 1 for w in result['weights']) 