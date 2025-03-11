# StockPulse: S&P 500 Market Performance Analyzer
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
import os
import glob
import json

# Set random seed for reproducibility
np.random.seed(42)

# Configure visualization settings
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')

# Create directory for output
if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not os.path.exists('data'):
    os.makedirs('data')


class StockAnalyzer:
    """
    A class for analyzing stock market performance using S&P 500 dataset from Kaggle.
    """
    
    def __init__(self, sector=None, top_n=None):
        """
        Initialize the StockAnalyzer.
        
        Parameters:
        -----------
        sector : str, optional
            Filter stocks by specific sector
        top_n : int, optional
            Number of top stocks by market cap to analyze
        """
        self.sector = sector
        self.top_n = top_n
        self.all_data = None
        self.tickers = None
        self.data = None
        self.returns = None
        self.stats = None
        self.models = {}
        self.sector_info = None
        
    def load_kaggle_data(self, data_path, all_stocks_file='all_stocks_5yr.csv', save=True):
        """
        Load S&P 500 stock data from Kaggle dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing the Kaggle dataset
        all_stocks_file : str, optional
            Filename of the CSV containing all stock data
        save : bool, optional
            Whether to save the processed data to CSV files
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the stock price data
        """
        print(f"Loading S&P 500 stock data from Kaggle dataset...")
        
        # Load the main stock data file
        file_path = os.path.join(data_path, all_stocks_file)
        self.all_data = pd.read_csv(file_path)
        
        # Convert date column to datetime
        self.all_data['date'] = pd.to_datetime(self.all_data['date'])
        
        # Set index
        self.all_data.set_index('date', inplace=True)
        
        # Load sector information if available
        try:
            sector_file = os.path.join(data_path, 'sectors.csv')
            self.sector_info = pd.read_csv(sector_file)
            print(f"Sector information loaded successfully.")
        except:
            print(f"No sector information file found. Proceeding without sector filtering.")
        
        # Get unique tickers
        all_tickers = self.all_data['Name'].unique()
        print(f"Found {len(all_tickers)} unique stocks in the dataset.")
        
        # Filter by sector if specified
        if self.returns is None:
            print("No returns data available. Please calculate returns first.")
            return None
        
        # Select ticker to plot
        if tickers is None:
            # Default to first ticker alphabetically
            plot_tickers = [sorted(self.returns.columns)[0]]
        else:
            # Validate provided tickers
            plot_tickers = [t for t in tickers if t in self.returns.columns]
            if not plot_tickers:
                print("None of the provided tickers were found in the data.")
                return None
        
        # Number of stocks to plot
        n_tickers = len(plot_tickers)
        
        # Create subplots
        fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 6 * n_tickers))
        
        # Handle case of a single ticker
        if n_tickers == 1:
            axes = [axes]
        
        for i, ticker in enumerate(plot_tickers):
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
                if drawdowns.iloc[j] < 0 and not in_drawdown:
                    # Start of a drawdown
                    in_drawdown = True
                    start_idx = j
                elif drawdowns.iloc[j] >= 0 and in_drawdown:
                    # End of a drawdown
                    in_drawdown = False
                    end_idx = j - 1
                    
                    # Calculate maximum drawdown in this period
                    max_drawdown = drawdowns.iloc[start_idx:end_idx+1].min()
                    
                    drawdown_periods.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'max_drawdown': max_drawdown
                    })
            
            # If we're still in a drawdown at the end
            if in_drawdown:
                end_idx = len(drawdowns) - 1
                max_drawdown = drawdowns.iloc[start_idx:end_idx+1].min()
                
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
                               xy=(mid_date, wealth_index.iloc[mid_point]),
                               xytext=(mid_date, wealth_index.iloc[mid_point] * 1.1),
                               arrowprops=dict(facecolor=colors[j % len(colors)], shrink=0.05),
                               fontsize=10)
            
            axes[i].set_title(f'Largest Drawdowns for {ticker}', fontsize=14)
            axes[i].set_xlabel('Date', fontsize=12)
            axes[i].set_ylabel('Wealth Index', fontsize=12)
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/sp500_largest_drawdowns.png', dpi=300)
            print("Largest drawdowns plot saved to 'outputs/sp500_largest_drawdowns.png'")
        
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
            print(f"No data available for {ticker}. Please load data first.")
            return None
        
        # Extract the price series for the ticker
        prices = self.data[ticker].copy()
        
        if model_type == 'linear':
            # Create features and target
            X = []
            y = []
            
            for i in range(feature_window, len(prices) - target_days):
                # Features: returns over past feature_window days
                feature_window_prices = prices.iloc[i-feature_window:i]
                feature_returns = feature_window_prices.pct_change().dropna().values
                
                if len(feature_returns) == feature_window - 1:  # Expected length
                    X.append(feature_returns)
                    
                    # Target: cumulative return over next target_days
                    future_return = (prices.iloc[i+target_days] / prices.iloc[i]) - 1
                    y.append(future_return)
            
            X = np.array(X)
            y = np.array(y)
            
            if len(X) == 0 or len(y) == 0:
                print(f"Insufficient data for {ticker} to build a model.")
                return None
            
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
            try:
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
                train_data = prices.iloc[:train_size]
                test_data = prices.iloc[train_size:]
                
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
                
            except Exception as e:
                print(f"Error building ARIMA model for {ticker}: {e}")
                return None
        
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
    
    def create_interactive_dashboard(self, tickers=None, save=True):
        """
        Create an interactive dashboard using Plotly.
        
        Parameters:
        -----------
        tickers : list, optional
            List of tickers to include (default: None - use top 5 by total return)
        save : bool, optional
            Whether to save the dashboard as HTML (default: True)
            
        Returns:
        --------
        plotly.graph_objects.Figure
            The created figure
        """
        if self.data is None or self.returns is None:
            print("No data available. Please load data and calculate returns first.")
            return None
        
        # Select tickers for the dashboard
        if tickers is None:
            # Use top 5 by total return
            total_returns = (self.data.iloc[-1] / self.data.iloc[0] - 1).sort_values(ascending=False)
            display_tickers = total_returns.head(5).index.tolist()
        else:
            display_tickers = [t for t in tickers if t in self.data.columns]
            if not display_tickers:
                print("None of the provided tickers were found in the data.")
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
                'Volume'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Colors for the tickers
        colors = px.colors.qualitative.Plotly[:len(display_tickers)]
        
        # 1. Normalized Stock Prices
        normalized_prices = self.data[display_tickers].copy() / self.data[display_tickers].iloc[0] * 100
        
        for i, ticker in enumerate(display_tickers):
            fig.add_trace(
                go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[ticker],
                    mode='lines',
                    name=f'{ticker} Price',
                    line=dict(width=2, color=colors[i])
                ),
                row=1, col=1
            )
        
        # 2. Cumulative Returns
        cumulative_returns = (1 + self.returns[display_tickers]).cumprod() - 1
        
        for i, ticker in enumerate(display_tickers):
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[ticker] * 100,
                    mode='lines',
                    name=f'{ticker} Cumulative',
                    line=dict(width=2, color=colors[i])
                ),
                row=2, col=1
            )
        
        # 3. Rolling Volatility
        window = 30
        for i, ticker in enumerate(display_tickers):
            rolling_vol = self.returns[ticker].rolling(window=window).std() * 100 * np.sqrt(252)  # Annualized
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    mode='lines',
                    name=f'{ticker} Volatility',
                    line=dict(width=2, color=colors[i])
                ),
                row=2, col=2
            )
        
        # 4. Rolling Mean Return
        for i, ticker in enumerate(display_tickers):
            rolling_mean = self.returns[ticker].rolling(window=window).mean() * 100 * 252  # Annualized
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_mean.index,
                    y=rolling_mean,
                    mode='lines',
                    name=f'{ticker} Mean',
                    line=dict(width=2, color=colors[i])
                ),
                row=3, col=1
            )
        
        # 5. Trading Volume
        for i, ticker in enumerate(display_tickers):
            # Filter volume data for this ticker
            ticker_volume_data = self.all_data[self.all_data['Name'] == ticker]['volume']
            
            if not ticker_volume_data.empty:
                # Resample to weekly for better visualization
                weekly_volume = ticker_volume_data.resample('W').sum()
                
                fig.add_trace(
                    go.Bar(
                        x=weekly_volume.index,
                        y=weekly_volume,
                        name=f'{ticker} Volume',
                        marker_color=colors[i]
                    ),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1200,
            title_text="S&P 500 Stock Performance Dashboard",
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
        fig.update_yaxes(title_text="Annualized Volatility (%)", row=2, col=2)
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Annualized Mean Return (%)", row=3, col=1)
        
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Weekly Volume", row=3, col=2)
        
        if save:
            fig.write_html('outputs/sp500_interactive_dashboard.html')
            print("Interactive dashboard saved to 'outputs/sp500_interactive_dashboard.html'")
        
        return figsector and self.sector_info is not None:
            sector_tickers = self.sector_info[self.sector_info['Sector'] == self.sector]['Symbol'].tolist()
            filtered_tickers = [t for t in all_tickers if t in sector_tickers]
            print(f"Filtered to {len(filtered_tickers)} stocks in {self.sector} sector.")
            self.tickers = filtered_tickers
        else:
            self.tickers = all_tickers
        
        # Limit to top N stocks by market cap if specified
        if self.top_n and self.top_n < len(self.tickers):
            if self.sector_info is not None and 'Market Cap' in self.sector_info.columns:
                # Get market cap data
                market_cap_data = self.sector_info[self.sector_info['Symbol'].isin(self.tickers)]
                market_cap_data = market_cap_data.sort_values('Market Cap', ascending=False)
                self.tickers = market_cap_data['Symbol'].head(self.top_n).tolist()
                print(f"Selected top {self.top_n} stocks by market cap.")
            else:
                # If no market cap data, use average volume as a proxy
                avg_volume = self.all_data.groupby('Name')['volume'].mean().sort_values(ascending=False)
                self.tickers = avg_volume.head(self.top_n).index.tolist()
                print(f"Selected top {self.top_n} stocks by average trading volume.")
        
        # Extract close prices for selected tickers
        self.process_price_data()
        
        # Save to CSV if requested
        if save:
            self.data.to_csv('data/sp500_prices.csv')
            print(f"Processed price data saved to 'data/sp500_prices.csv'")
        
        return self.data

    def process_price_data(self):
        """
        Process raw data to extract close prices for selected tickers.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing close prices for selected tickers
        """
        # Filter for selected tickers
        filtered_data = self.all_data[self.all_data['Name'].isin(self.tickers)]
        
        # Pivot to get close prices for each ticker
        pivoted_data = filtered_data.pivot(columns='Name', values='close')
        
        # Handle missing data
        pivoted_data = pivoted_data.fillna(method='ffill')
        
        # Keep only tickers with sufficient data (at least 80% non-NaN values)
        min_records = 0.8 * len(pivoted_data)
        sufficient_data_tickers = [ticker for ticker in pivoted_data.columns 
                                  if pivoted_data[ticker].count() >= min_records]
        
        if len(sufficient_data_tickers) < len(pivoted_data.columns):
            print(f"Removed {len(pivoted_data.columns) - len(sufficient_data_tickers)} tickers with insufficient data.")
        
        # Final dataset
        self.data = pivoted_data[sufficient_data_tickers]
        self.tickers = sufficient_data_tickers
        
        return self.data
        
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
            print("No data available. Please load data first.")
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
            self.returns.to_csv(f'data/sp500_{period}_returns.csv')
            print(f"{period.capitalize()} returns saved to 'data/sp500_{period}_returns.csv'")
        
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
            self.stats.to_csv('data/sp500_statistics.csv')
            print("Statistics saved to 'data/sp500_statistics.csv'")
        
        return self.stats
    
    def plot_prices(self, normalize=True, top_n=10, save=True):
        """
        Plot stock prices over time.
        
        Parameters:
        -----------
        normalize : bool, optional
            Whether to normalize prices to start at 100 (default: True)
        top_n : int, optional
            Number of top performing stocks to highlight (default: 10)
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.data is None:
            print("No data available. Please load data first.")
            return None
        
        # If there are too many tickers, select top performers
        if len(self.data.columns) > top_n:
            # Calculate total return for each stock
            total_returns = (self.data.iloc[-1] / self.data.iloc[0]) - 1
            top_performers = total_returns.nlargest(top_n).index
            plot_data = self.data[top_performers].copy()
            title_prefix = f"Top {top_n} Performing S&P 500 Stocks"
        else:
            plot_data = self.data.copy()
            title_prefix = "S&P 500 Stocks"
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize data if requested
        if normalize:
            plot_data = plot_data / plot_data.iloc[0] * 100
            title = f'{title_prefix} - Normalized Price (Base = 100)'
            ylabel = 'Normalized Price'
        else:
            title = f'{title_prefix} - Price Over Time'
            ylabel = 'Price ($)'
        
        # Plot each ticker
        for column in plot_data.columns:
            ax.plot(plot_data.index, plot_data[column], linewidth=2, label=column)
        
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/sp500_prices.png', dpi=300)
            print("Stock price plot saved to 'outputs/sp500_prices.png'")
        
        return fig
    
    def plot_returns_distribution(self, top_n=6, save=True):
        """
        Plot the distribution of returns for selected stocks.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of stocks to plot (default: 6)
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
        
        # Select stocks for plotting
        if len(self.returns.columns) > top_n:
            # Use stocks with highest volatility for more interesting distributions
            volatility = self.returns.std().nlargest(top_n)
            selected_tickers = volatility.index
        else:
            selected_tickers = self.returns.columns
        
        # Create figure with subplots
        n_tickers = len(selected_tickers)
        fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 4 * n_tickers))
        
        # Handle case of a single ticker
        if n_tickers == 1:
            axes = [axes]
        
        # Plot distribution for each ticker
        for i, ticker in enumerate(selected_tickers):
            sns.histplot(self.returns[ticker].dropna(), kde=True, ax=axes[i])
            
            # Add vertical line for mean
            mean_return = self.returns[ticker].mean()
            axes[i].axvline(mean_return, color='r', linestyle='--', 
                         label=f'Mean: {mean_return:.4f}')
            
            # Add normal distribution overlay
            from scipy import stats as scistat
            x = np.linspace(self.returns[ticker].min(), self.returns[ticker].max(), 100)
            mu, std = scistat.norm.fit(self.returns[ticker].dropna())
            p = scistat.norm.pdf(x, mu, std)
            axes[i].plot(x, p * len(self.returns[ticker]) * (x[1]-x[0]), 'k--', linewidth=1.5, 
                       label='Normal Dist.')
            
            axes[i].set_title(f'Return Distribution for {ticker}', fontsize=14)
            axes[i].set_xlabel('Return', fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/sp500_return_distributions.png', dpi=300)
            print("Return distributions plot saved to 'outputs/sp500_return_distributions.png'")
        
        return fig
    
    def plot_correlation_matrix(self, min_corr=None, save=True):
        """
        Plot the correlation matrix of stock returns.
        
        Parameters:
        -----------
        min_corr : float, optional
            Minimum correlation to display (default: None - show all)
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
        
        # Filter correlations if requested
        if min_corr is not None:
            # Create mask for low correlations
            corr_mask = np.abs(corr_matrix) < min_corr
            # Replace low correlations with NaN
            corr_filtered = corr_matrix.mask(corr_mask)
        else:
            corr_filtered = corr_matrix
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_filtered, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Plot heatmap
        ax = sns.heatmap(corr_filtered, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 8})
        
        # If there are many stocks, don't show annotations to avoid clutter
        if len(corr_filtered) <= 15:
            ax = sns.heatmap(corr_filtered, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                       square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", annot_kws={"size": 8})
        
        plt.title('Correlation Matrix of Stock Returns', fontsize=16)
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/sp500_correlation_matrix.png', dpi=300)
            print("Correlation matrix plot saved to 'outputs/sp500_correlation_matrix.png'")
        
        return plt.gcf()
    
    def plot_sector_performance(self, save=True):
        """
        Plot performance by sector if sector data is available.
        
        Parameters:
        -----------
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.sector_info is None:
            print("No sector information available.")
            return None
        
        if self.data is None:
            print("No price data available. Please load data first.")
            return None
        
        # Calculate total returns for all stocks
        start_date = self.data.index.min()
        end_date = self.data.index.max()
        
        # Get prices at the start and end dates
        start_prices = self.data.loc[start_date:].iloc[0]
        end_prices = self.data.loc[:end_date].iloc[-1]
        
        # Calculate total return
        total_returns = (end_prices / start_prices) - 1
        
        # Create a DataFrame with ticker and return
        returns_df = pd.DataFrame({
            'Ticker': total_returns.index,
            'Total_Return': total_returns.values * 100  # Convert to percentage
        })
        
        # Merge with sector information
        returns_with_sector = returns_df.merge(
            self.sector_info[['Symbol', 'Sector']], 
            left_on='Ticker', 
            right_on='Symbol',
            how='inner'
        )
        
        # Calculate average return by sector
        sector_returns = returns_with_sector.groupby('Sector')['Total_Return'].agg(['mean', 'std', 'count'])
        sector_returns = sector_returns.sort_values('mean', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot sector returns with error bars
        sector_returns_plot = sector_returns[sector_returns['count'] >= 5]  # At least 5 stocks in sector
        
        ax.barh(sector_returns_plot.index, sector_returns_plot['mean'], 
               xerr=sector_returns_plot['std'], alpha=0.7, color='skyblue')
        
        # Add number of stocks in each sector
        for i, (sector, row) in enumerate(sector_returns_plot.iterrows()):
            ax.text(row['mean'] + 5, i, f"n={int(row['count'])}", va='center')
        
        ax.set_title(f'S&P 500 Sector Performance ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})', 
                    fontsize=15)
        ax.set_xlabel('Total Return (%)', fontsize=12)
        ax.set_ylabel('Sector', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/sp500_sector_performance.png', dpi=300)
            print("Sector performance plot saved to 'outputs/sp500_sector_performance.png'")
        
        return fig
    
    def plot_rolling_statistics(self, tickers=None, window=30, save=True):
        """
        Plot rolling mean and volatility for selected stocks.
        
        Parameters:
        -----------
        tickers : list, optional
            List of tickers to plot (default: None - use top 5 by volatility)
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
        
        # Select tickers to plot
        if tickers is None:
            # Use top 5 by volatility
            vol_rank = self.returns.std().sort_values(ascending=False)
            plot_tickers = vol_rank.head(5).index.tolist()
        else:
            # Validate provided tickers
            plot_tickers = [t for t in tickers if t in self.returns.columns]
            if not plot_tickers:
                print("None of the provided tickers were found in the data.")
                return None
        
        # Create subplots
        n_tickers = len(plot_tickers)
        fig, axes = plt.subplots(n_tickers, 2, figsize=(16, 5 * n_tickers))
        
        # Handle case of a single ticker
        if n_tickers == 1:
            axes = [axes]
        
        for i, ticker in enumerate(plot_tickers):
            # Rolling mean
            rolling_mean = self.returns[ticker].rolling(window=window).mean() * 100  # Convert to percentage
            axes[i, 0].plot(self.returns.index, rolling_mean, linewidth=2)
            axes[i, 0].set_title(f'{ticker} - {window}-Day Rolling Mean Return', fontsize=14)
            axes[i, 0].set_xlabel('Date', fontsize=12)
            axes[i, 0].set_ylabel('Mean Return (%)', fontsize=12)
            
            # Rolling volatility
            rolling_vol = self.returns[ticker].rolling(window=window).std() * 100 * np.sqrt(252)  # Annualized
            axes[i, 1].plot(self.returns.index, rolling_vol, linewidth=2, color='orange')
            axes[i, 1].set_title(f'{ticker} - {window}-Day Rolling Annualized Volatility', fontsize=14)
            axes[i, 1].set_xlabel('Date', fontsize=12)
            axes[i, 1].set_ylabel('Annualized Volatility (%)', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'outputs/sp500_rolling_statistics_{window}d.png', dpi=300)
            print(f"Rolling statistics plot saved to 'outputs/sp500_rolling_statistics_{window}d.png'")
        
        return fig
    
    def plot_cumulative_returns(self, top_n=10, save=True):
        """
        Plot cumulative returns for top performing stocks.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top stocks to show (default: 10)
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
        
        # Get top performers at the end
        final_returns = cumulative_returns.iloc[-1].sort_values(ascending=False)
        top_performers = final_returns.head(top_n).index
        
        # Plot cumulative returns for top performers
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for ticker in top_performers:
            ax.plot(cumulative_returns.index, cumulative_returns[ticker] * 100, 
                   linewidth=2, label=ticker)
        
        ax.set_title(f'Cumulative Returns - Top {top_n} Performers', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('outputs/sp500_cumulative_returns.png', dpi=300)
            print("Cumulative returns plot saved to 'outputs/sp500_cumulative_returns.png'")
        
        return fig
    
    def analyze_drawdowns(self, tickers=None, num_drawdowns=5, save=True):
        """
        Analyze the largest drawdowns for selected stocks.
        
        Parameters:
        -----------
        tickers : list, optional
            List of tickers to analyze (default: None - use top 5 by market cap)
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
        
        # Select tickers to analyze
        if tickers is None:
            # Default to top 5 by average volume as a proxy for market cap
            volumes = self.all_data.groupby('Name')['volume'].mean().sort_values(ascending=False)
            analyze_tickers = volumes.head(5).index.tolist()
            analyze_tickers = [t for t in analyze_tickers if t in self.returns.columns]
        else:
            # Validate provided tickers
            analyze_tickers = [t for t in tickers if t in self.returns.columns]
            if not analyze_tickers:
                print("None of the provided tickers were found in the data.")
                return None
        
        drawdown_results = {}
        
        for ticker in analyze_tickers:
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
                if is_in_drawdown.iloc[i] and not in_drawdown:
                    # Start of a new drawdown
                    current_drawdown += 1
                    in_drawdown = True
                elif not is_in_drawdown.iloc[i] and in_drawdown:
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
                    while (recovery_idx < len(drawdowns) and 
                           wealth_index.iloc[recovery_idx] < previous_peaks.iloc[np.where(period_mask)[0][0]]):
                        recovery_idx += 1
                        if recovery_idx >= len(drawdowns):
                            break
                    
                    if recovery_idx < len(drawdowns):
                        recovery_date = drawdowns.index[recovery_idx]
                
                duration = (end_date - start_date).days
                recovery_duration = (recovery_date - end_date).days if recovery_date else None
                
                drawdown_info.append({
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'max_drawdown': float(max_drawdown * 100),  # Convert to percentage
                    'max_drawdown_date': max_drawdown_date.strftime('%Y-%m-%d'),
                    'recovery_date': recovery_date.strftime('%Y-%m-%d') if recovery_date else None,
                    'duration_days': duration,
                    'recovery_duration_days': recovery_duration
                })
            
            # Sort drawdowns by magnitude
            drawdown_info = sorted(drawdown_info, key=lambda x: x['max_drawdown'])
            
            # Take the largest drawdowns
            drawdown_results[ticker] = drawdown_info[:num_drawdowns]
        
        if save:
            # Save as JSON
            with open('outputs/sp500_drawdown_analysis.json', 'w') as f:
                json.dump(drawdown_results, f, indent=4)
            print("Drawdown analysis saved to 'outputs/sp500_drawdown_analysis.json'")
        
        return drawdown_results
    
    def plot_drawdowns(self, tickers=None, num_drawdowns=3, save=True):
        """
        Plot the largest drawdowns for selected stocks.
        
        Parameters:
        -----------
        tickers : list, optional
            List of tickers to plot (default: None - use first ticker by alphabet)
        num_drawdowns : int, optional
            Number of largest drawdowns to plot (default: 3)
        save : bool, optional
            Whether to save the plot (default: True)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.