# Stock Market Performance Analysis
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set random seed for reproducibility
np.random.seed(42)

# Configure visualization settings
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')

# Create directory for output
import os
if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not os.path.exists('data'):
    os.makedirs('data')


class StockAnalyzer:
    """
    A class for analyzing stock market performance using historical data.
    """
    
    def __init__(self, tickers, start_date, end_date=None):
        """
        Initialize the StockAnalyzer with ticker symbols and date range.
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format (default: today)
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.stats = None
        self.models = {}
        
    def fetch_data(self, save=True):
        """
        Fetch historical stock data using yfinance.
        
        Parameters:
        -----------
        save : bool, optional
            Whether to save the data to a CSV file (default: True)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the stock price data
        """
        print(f"Fetching stock data for {self.tickers} from {self.start_date} to {self.end_date}...")
        
        # Fetch data for each ticker
        data = {}
        for ticker in self.tickers:
            try:
                stock_data = yf.download(ticker, start=self.start_date, end=self.end_date)
                if stock_data.empty:
                    print(f"No data found for {ticker}")
                    continue
                data[ticker] = stock_data['Adj Close']
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        # Combine all tickers into a single DataFrame
        if data:
            self.data = pd.DataFrame(data)
            self.data.index.name = 'Date'
            
            # Save to CSV
            if save:
                self.data.to_csv('data/stock_prices.csv')
                print(f"Data saved to 'data/stock_prices.csv'")
            
            return self.data
        else:
            print("No data fetched. Please check ticker symbols and date range.")
            return None
        
    def load_data(self, file_path):
        """
        Load stock data from a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the stock price data
        """
        try:
            self.data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            print(f"Data loaded from '{file_path}'")
            return self.data
        except Exception as e:
            print(f"Error loading data from '{file_path}': {e}")
            return None
        
    def calculate_returns(self, period='daily', save=True):
        """
        Calculate stock returns.
        
        Parameters:
        -----------
        period : str, optional
            The period for return calculation - 'daily', 'weekly', 'monthly' (default: 'daily')
        save : bool, optional
            Whether to save returns to a CSV file (default: True)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the returns data
        """
        if self.data is None:
            print("No data available. Please fetch or load data first.")
            return None
        
        # Resample data if needed
        data = self.data.copy()
        if period == 'weekly':
            data = self.data.resample('W').last()
        elif period == 'monthly':
            data = self.data.resample('M').last()
        
        # Calculate returns
        self.returns = data.pct_change().dropna()
        
        if save:
            self.returns.to_csv(f'data/{period}_returns.csv')
            print(f"{period.capitalize()} returns saved to 'data/{period}_returns.csv'")
        
        return self.returns
    
    def calculate_statistics(self, annualize=True, save=True):
        """
        Calculate key statistics for each stock.
        
        Parameters:
        -----------
        annualize : bool, optional
            Whether to annualize statistics (default: True)
        save : bool, optional
            Whether to save statistics to a CSV file (default: True)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the calculated statistics
        """
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return None
        
        # Calculate statistics
        stats = {}
        
        # Mean return
        stats['Mean Return'] = self.returns.mean()
        
        # Standard deviation (volatility)
        stats['Volatility'] = self.returns.std()
        
        # Annualize if needed
        if annualize:
            # Assuming 252 trading days, 52 weeks, or 12 months
            if len(self.returns) / len(self.data) > 0.9:  # daily returns
                factor = 252
            elif len(self.returns) / len(self.data) > 0.15:  # weekly returns
                factor = 52
            else:  # monthly returns
                factor = 12
                
            stats['Mean Return'] *= factor
            stats['Volatility'] *= np.sqrt(factor)
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        stats['Sharpe Ratio'] = stats['Mean Return'] / stats['Volatility']
        
        # Maximum drawdown
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        stats['Max Drawdown'] = drawdown.min()
        
        # Convert to DataFrame
        self.stats = pd.DataFrame(stats)
        
        if save:
            self.stats.to_csv('data/stock_statistics.csv')
            print("Statistics saved to 'data/stock_statistics.csv'")
        
        return self.stats
    
    def plot_prices(self, normalize=True, save=True):
        """
        Plot stock prices over time.
        
        Parameters:
        -----------
        normalize : bool, optional
            Whether to normalize prices to start at 100 (default: True)
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.data is None:
            print("No data available. Please fetch or load data first.")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize data if requested
        if normalize:
            data = self.data.copy() / self.data.iloc[0] * 100
            title = 'Normalized Stock Prices (Base = 100)'
            ylabel = 'Normalized Price'
        else:
            data = self.data.copy()
            title = 'Stock Prices Over Time'
            ylabel = 'Price ($)'
        
        # Plot each ticker
        for column in data.columns:
            ax.plot(data.index, data[column], linewidth=2, label=column)
        
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/stock_prices.png', dpi=300)
            print("Stock price plot saved to 'outputs/stock_prices.png'")
        
        return fig
    
    def plot_returns_distribution(self, save=True):
        """
        Plot the distribution of returns for each stock.
        
        Parameters:
        -----------
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return None
        
        # Create figure with subplots
        n_tickers = len(self.returns.columns)
        fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 5 * n_tickers))
        
        # Handle case of a single ticker
        if n_tickers == 1:
            axes = [axes]
        
        # Plot distribution for each ticker
        for i, ticker in enumerate(self.returns.columns):
            sns.histplot(self.returns[ticker], kde=True, ax=axes[i])
            
            # Add vertical line for mean
            mean_return = self.returns[ticker].mean()
            axes[i].axvline(mean_return, color='r', linestyle='--', 
                         label=f'Mean: {mean_return:.4f}')
            
            axes[i].set_title(f'Return Distribution for {ticker}', fontsize=14)
            axes[i].set_xlabel('Return', fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/return_distributions.png', dpi=300)
            print("Return distributions plot saved to 'outputs/return_distributions.png'")
        
        return fig
    
    def plot_correlation_matrix(self, save=True):
        """
        Plot the correlation matrix of stock returns.
        
        Parameters:
        -----------
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return None
        
        # Calculate correlation matrix
        corr_matrix = self.returns.corr()
        
        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f")
        
        ax.set_title('Correlation Matrix of Stock Returns', fontsize=15)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/correlation_matrix.png', dpi=300)
            print("Correlation matrix plot saved to 'outputs/correlation_matrix.png'")
        
        return fig
    
    def plot_rolling_statistics(self, window=30, save=True):
        """
        Plot rolling mean and volatility for each stock.
        
        Parameters:
        -----------
        window : int, optional
            Rolling window size in days (default: 30)
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return None
        
        # Create subplots
        n_tickers = len(self.returns.columns)
        fig, axes = plt.subplots(n_tickers, 2, figsize=(16, 5 * n_tickers))
        
        # Handle case of a single ticker
        if n_tickers == 1:
            axes = [axes]
        
        for i, ticker in enumerate(self.returns.columns):
            # Rolling mean
            rolling_mean = self.returns[ticker].rolling(window=window).mean()
            axes[i, 0].plot(self.returns.index, rolling_mean, linewidth=2)
            axes[i, 0].set_title(f'{ticker} - {window}-Day Rolling Mean Return', fontsize=14)
            axes[i, 0].set_xlabel('Date', fontsize=12)
            axes[i, 0].set_ylabel('Mean Return', fontsize=12)
            
            # Rolling volatility
            rolling_vol = self.returns[ticker].rolling(window=window).std()
            axes[i, 1].plot(self.returns.index, rolling_vol, linewidth=2, color='orange')
            axes[i, 1].set_title(f'{ticker} - {window}-Day Rolling Volatility', fontsize=14)
            axes[i, 1].set_xlabel('Date', fontsize=12)
            axes[i, 1].set_ylabel('Volatility', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'outputs/rolling_statistics_{window}d.png', dpi=300)
            print(f"Rolling statistics plot saved to 'outputs/rolling_statistics_{window}d.png'")
        
        return fig
    
    def plot_cumulative_returns(self, save=True):
        """
        Plot cumulative returns for each stock.
        
        Parameters:
        -----------
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return None
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.returns).cumprod() - 1
        
        # Plot cumulative returns
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for column in cumulative_returns.columns:
            ax.plot(cumulative_returns.index, cumulative_returns[column] * 100, 
                   linewidth=2, label=column)
        
        ax.set_title('Cumulative Returns Over Time', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/cumulative_returns.png', dpi=300)
            print("Cumulative returns plot saved to 'outputs/cumulative_returns.png'")
        
        return fig
    
    def analyze_drawdowns(self, num_drawdowns=5, save=True):
        """
        Analyze the largest drawdowns for each stock.
        
        Parameters:
        -----------
        num_drawdowns : int, optional
            Number of largest drawdowns to analyze (default: 5)
        save : bool, optional
            Whether to save the results (default: True)
            
        Returns:
        --------
        dict
            Dictionary containing drawdown analysis for each ticker
        """
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return None
        
        drawdown_results = {}
        
        for ticker in self.returns.columns:
            # Calculate wealth index (cumulative returns)
            wealth_index = (1 + self.returns[ticker]).cumprod()
            
            # Calculate previous peaks
            previous_peaks = wealth_index.cummax()
            
            # Calculate drawdowns
            drawdowns = (wealth_index / previous_peaks) - 1
            
            # Find drawdown periods
            is_in_drawdown = drawdowns < 0
            
            # Label drawdown periods
            drawdown_labels = np.zeros_like(drawdowns, dtype=int)
            current_drawdown = 0
            in_drawdown = False
            
            for i in range(len(drawdowns)):
                if is_in_drawdown[i] and not in_drawdown:
                    # Start of a new drawdown
                    current_drawdown += 1
                    in_drawdown = True
                elif not is_in_drawdown[i] and in_drawdown:
                    # End of a drawdown
                    in_drawdown = False
                
                if in_drawdown:
                    drawdown_labels[i] = current_drawdown
            
            # Collect information about each drawdown
            drawdown_info = []
            
            for i in range(1, current_drawdown + 1):
                period_mask = drawdown_labels == i
                if not any(period_mask):
                    continue
                
                period_drawdowns = drawdowns[period_mask]
                if len(period_drawdowns) == 0:
                    continue
                
                start_date = drawdowns.index[period_mask][0]
                end_date = drawdowns.index[period_mask][-1]
                max_drawdown = period_drawdowns.min()
                max_drawdown_date = period_drawdowns.idxmin()
                recovery_date = None
                
                # Find recovery date if available
                if end_date != drawdowns.index[-1]:
                    recovery_idx = np.where(period_mask)[0][-1] + 1
                    while recovery_idx < len(drawdowns) and wealth_index[recovery_idx] < previous_peaks[np.where(period_mask)[0][0]]:
                        recovery_idx += 1
                        if recovery_idx >= len(drawdowns):
                            break
                    
                    if recovery_idx < len(drawdowns):
                        recovery_date = drawdowns.index[recovery_idx]
                
                duration = (end_date - start_date).days
                recovery_duration = (recovery_date - end_date).days if recovery_date else None
                
                drawdown_info.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'max_drawdown': max_drawdown * 100,  # Convert to percentage
                    'max_drawdown_date': max_drawdown_date,
                    'recovery_date': recovery_date,
                    'duration_days': duration,
                    'recovery_duration_days': recovery_duration
                })
            
            # Sort drawdowns by magnitude
            drawdown_info = sorted(drawdown_info, key=lambda x: x['max_drawdown'])
            
            # Take the largest drawdowns
            drawdown_results[ticker] = drawdown_info[:num_drawdowns]
        
        if save:
            # Save as JSON
            import json
            with open('outputs/drawdown_analysis.json', 'w') as f:
                json.dump(drawdown_results, f, indent=4, default=str)
            print("Drawdown analysis saved to 'outputs/drawdown_analysis.json'")
        
        return drawdown_results
    
    def plot_drawdowns(self, num_drawdowns=3, save=True):
        """
        Plot the largest drawdowns for each stock.
        
        Parameters:
        -----------
        num_drawdowns : int, optional
            Number of largest drawdowns to plot (default: 3)
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return None
        
        # Number of stocks to plot
        n_tickers = len(self.returns.columns)
        
        # Create subplots
        fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 6 * n_tickers))
        
        # Handle case of a single ticker
        if n_tickers == 1:
            axes = [axes]
        
        for i, ticker in enumerate(self.returns.columns):
            # Calculate wealth index
            wealth_index = (1 + self.returns[ticker]).cumprod()
            
            # Calculate previous peaks
            previous_peaks = wealth_index.cummax()
            
            # Calculate drawdowns
            drawdowns = (wealth_index / previous_peaks) - 1
            
            # Plot wealth index and peaks
            axes[i].plot(wealth_index.index, wealth_index, label='Wealth Index', linewidth=2)
            axes[i].plot(previous_peaks.index, previous_peaks, label='Previous Peak', 
                       linestyle='--', linewidth=1, color='green')
            
            # Find the largest drawdowns
            drawdown_periods = []
            in_drawdown = False
            start_idx = None
            
            for j in range(len(drawdowns)):
                if drawdowns[j] < 0 and not in_drawdown:
                    # Start of a drawdown
                    in_drawdown = True
                    start_idx = j
                elif drawdowns[j] >= 0 and in_drawdown:
                    # End of a drawdown
                    in_drawdown = False
                    end_idx = j - 1
                    
                    # Calculate maximum drawdown in this period
                    max_drawdown = drawdowns[start_idx:end_idx+1].min()
                    
                    drawdown_periods.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'max_drawdown': max_drawdown
                    })
            
            # If we're still in a drawdown at the end
            if in_drawdown:
                end_idx = len(drawdowns) - 1
                max_drawdown = drawdowns[start_idx:end_idx+1].min()
                
                drawdown_periods.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'max_drawdown': max_drawdown
                })
            
            # Sort by drawdown magnitude and take the largest
            drawdown_periods.sort(key=lambda x: x['max_drawdown'])
            largest_drawdowns = drawdown_periods[:num_drawdowns]
            
            # Plot the largest drawdowns
            colors = ['red', 'orange', 'purple']
            for j, dd in enumerate(largest_drawdowns):
                start_date = drawdowns.index[dd['start_idx']]
                end_date = drawdowns.index[dd['end_idx']]
                
                # Highlight drawdown period
                axes[i].axvspan(start_date, end_date, alpha=0.2, color=colors[j % len(colors)])
                
                # Add text annotation
                mid_point = dd['start_idx'] + (dd['end_idx'] - dd['start_idx']) // 2
                mid_date = drawdowns.index[mid_point]
                
                axes[i].annotate(f"Drawdown: {dd['max_drawdown']*100:.1f}%", 
                               xy=(mid_date, wealth_index[mid_point]),
                               xytext=(mid_date, wealth_index[mid_point] * 1.1),
                               arrowprops=dict(facecolor=colors[j % len(colors)], shrink=0.05),
                               fontsize=10)
            
            axes[i].set_title(f'Largest Drawdowns for {ticker}', fontsize=14)
            axes[i].set_xlabel('Date', fontsize=12)
            axes[i].set_ylabel('Wealth Index', fontsize=12)
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/largest_drawdowns.png', dpi=300)
            print("Largest drawdowns plot saved to 'outputs/largest_drawdowns.png'")
        
        return fig
    
    def build_prediction_model(self, ticker, feature_window=30, target_days=5, model_type='linear'):
        """
        Build a prediction model for a specific stock.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to build a model for
        feature_window : int, optional
            Number of previous days to use as features (default: 30)
        target_days : int, optional
            Number of days ahead to predict (default: 5)
        model_type : str, optional
            Type of model to build - 'linear' or 'arima' (default: 'linear')
            
        Returns:
        --------
        tuple
            Model object and evaluation metrics
        """
        if self.data is None or ticker not in self.data.columns:
            print(f"No data available for {ticker}. Please fetch or load data first.")
            return None
        
        # Extract the price series for the ticker
        prices = self.data[ticker].copy()
        
        if model_type == 'linear':
            # Create features and target
            X = []
            y = []
            
            for i in range(feature_window, len(prices) - target_days):
                # Features: returns over past feature_window days
                X.append(prices[i-feature_window:i].pct_change().dropna().values)
                
                # Target: cumulative return over next target_days
                future_return = (prices[i+target_days] / prices[i]) - 1
                y.append(future_return)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Build linear regression model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model components
            self.models[ticker] = {
                'model': model,
                'scaler': scaler,
                'feature_window': feature_window,
                'target_days': target_days,
                'type': 'linear',
                'metrics': {
                    'mse': mse,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                },
                'test_data': {
                    'y_test': y_test,
                    'y_pred': y_pred
                }
            }
            
            print(f"Linear model for {ticker} built successfully.")
            print(f"Model metrics - RMSE: {np.sqrt(mse):.6f}, R²: {r2:.6f}")
            
            return model, {'mse': mse, 'r2': r2, 'rmse': np.sqrt(mse)}
        
        elif model_type == 'arima':
            # Use statsmodels ARIMA
            # Auto-determine p, d, q parameters
            from pmdarima import auto_arima
            
            # Use auto_arima to find best parameters
            auto_model = auto_arima(prices,
                                   start_p=1, start_q=1,
                                   max_p=3, max_q=3,
                                   d=1,  # Typically 1 for financial data
                                   seasonal=False,
                                   trace=True,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)
            
            # Get the optimal parameters
            p, d, q = auto_model.order
            
            # Split data for training and testing
            train_size = int(len(prices) * 0.8)
            train_data = prices[:train_size]
            test_data = prices[train_size:]
            
            # Build ARIMA model with optimal parameters
            model = ARIMA(train_data, order=(p, d, q))
            model_fit = model.fit()
            
            # Forecast
            forecast_steps = len(test_data)
            forecast = model_fit.forecast(steps=forecast_steps)
            
            # Evaluate
            mse = mean_squared_error(test_data, forecast)
            rmse = np.sqrt(mse)
            
            # Store model components
            self.models[ticker] = {
                'model': model_fit,
                'type': 'arima',
                'order': (p, d, q),
                'metrics': {
                    'mse': mse,
                    'rmse': rmse
                },
                'test_data': {
                    'actual': test_data,
                    'forecast': forecast
                }
            }
            
            print(f"ARIMA model for {ticker} built successfully.")
            print(f"Model parameters (p, d, q): ({p}, {d}, {q})")
            print(f"Model metrics - RMSE: {rmse:.6f}")
            
            return model_fit, {'mse': mse, 'rmse': rmse}
        
        else:
            print(f"Unknown model type: {model_type}. Please use 'linear' or 'arima'.")
            return None
    
    def plot_model_performance(self, ticker, save=True):
        """
        Plot actual vs. predicted values for a model.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to plot model performance for
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if ticker not in self.models:
            print(f"No model found for {ticker}. Please build a model first.")
            return None
        
        model_info = self.models[ticker]
        model_type = model_info['type']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if model_type == 'linear':
            # Extract test data
            y_test = model_info['test_data']['y_test']
            y_pred = model_info['test_data']['y_pred']
            
            # Create index for x-axis
            x = np.arange(len(y_test))
            
            # Plot actual vs predicted
            ax.plot(x, y_test, label='Actual Returns', linewidth=2)
            ax.plot(x, y_pred, label='Predicted Returns', linewidth=2, linestyle='--')
            
            # Add zero line
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Calculate correlation
            correlation = np.corrcoef(y_test, y_pred)[0, 1]
            
            ax.set_title(f'{ticker} - {model_info["target_days"]}-Day Return Prediction (R: {correlation:.3f})', fontsize=15)
            ax.set_xlabel('Test Sample Index', fontsize=12)
            ax.set_ylabel(f'{model_info["target_days"]}-Day Return', fontsize=12)
            ax.legend()
            ax.grid(True)
            
        elif model_type == 'arima':
            # Extract test data
            actual = model_info['test_data']['actual']
            forecast = model_info['test_data']['forecast']
            
            # Plot actual vs forecast
            ax.plot(actual.index, actual, label='Actual Prices', linewidth=2)
            ax.plot(forecast.index, forecast, label='Forecasted Prices', linewidth=2, linestyle='--')
            
            # Add model order to title
            p, d, q = model_info['order']
            
            ax.set_title(f'{ticker} - ARIMA({p},{d},{q}) Price Forecast', fontsize=15)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'outputs/{ticker}_model_performance.png', dpi=300)
            print(f"Model performance plot saved to 'outputs/{ticker}_model_performance.png'")
        
        return fig
    
    def create_interactive_dashboard(self, save=True):
        """
        Create an interactive dashboard using Plotly.
        
        Parameters:
        -----------
        save : bool, optional
            Whether to save the dashboard as HTML (default: True)
            
        Returns:
        --------
        plotly.graph_objects.Figure
            The created figure
        """
        if self.data is None or self.returns is None:
            print("No data available. Please fetch or load data and calculate returns first.")
            return None
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Stock Prices Over Time (Normalized)',
                'Cumulative Returns',
                'Rolling 30-Day Volatility',
                'Rolling 30-Day Mean Return',
                'Return Distributions',
                'Correlation Heatmap'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # 1. Normalized Stock Prices
        normalized_prices = self.data.copy() / self.data.iloc[0] * 100
        
        for ticker in normalized_prices.columns:
            fig.add_trace(
                go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[ticker],
                    mode='lines',
                    name=f'{ticker} Price',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 2. Cumulative Returns
        cumulative_returns = (1 + self.returns).cumprod() - 1
        
        for ticker in cumulative_returns.columns:
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[ticker] * 100,
                    mode='lines',
                    name=f'{ticker} Cumulative',
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # 3. Rolling Volatility
        window = 30
        for ticker in self.returns.columns:
            rolling_vol = self.returns[ticker].rolling(window=window).std() * 100
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    mode='lines',
                    name=f'{ticker} Volatility',
                    line=dict(width=2)
                ),
                row=2, col=2
            )
        
        # 4. Rolling Mean Return
        for ticker in self.returns.columns:
            rolling_mean = self.returns[ticker].rolling(window=window).mean() * 100
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_mean.index,
                    y=rolling_mean,
                    mode='lines',
                    name=f'{ticker} Mean',
                    line=dict(width=2)
                ),
                row=3, col=1
            )
        
        # 5. Correlation Heatmap
        corr_matrix = self.returns.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                colorbar=dict(title='Correlation')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1200,
            title_text="Stock Market Performance Dashboard",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price (Normalized to 100)", row=1, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="30-Day Rolling Volatility (%)", row=2, col=2)
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="30-Day Rolling Mean Return (%)", row=3, col=1)
        
        if save:
            fig.write_html('outputs/interactive_dashboard.html')
            print("Interactive dashboard saved to 'outputs/interactive_dashboard.html'")
        
        return fig


# Example Usage Script
if __name__ == "__main__":
    # Define tickers to analyze
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Define date range (5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Initialize analyzer
    analyzer = StockAnalyzer(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Fetch data
    data = analyzer.fetch_data()
    
    # Calculate returns
    returns = analyzer.calculate_returns()
    
    # Calculate statistics
    stats = analyzer.calculate_statistics()
    print("\nStock Statistics:")
    print(stats)
    
    # Generate visualizations
    analyzer.plot_prices()
    analyzer.plot_returns_distribution()
    analyzer.plot_correlation_matrix()
    analyzer.plot_rolling_statistics()
    analyzer.plot_cumulative_returns()
    analyzer.plot_drawdowns()
    
    # Analyze drawdowns
    drawdowns = analyzer.analyze_drawdowns()
    
    # Build prediction models for each stock
    for ticker in tickers:
        analyzer.build_prediction_model(ticker, feature_window=30, target_days=5, model_type='linear')
        analyzer.plot_model_performance(ticker)
    
    # Create interactive dashboard
    analyzer.create_interactive_dashboard()
    
    print("\nAnalysis completed! Check the 'outputs' directory for visualizations.")