#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating the use of StockAnalyzer class.
This script performs a complete analysis on a set of tech stocks.
"""

from datetime import datetime, timedelta
from stock_analyzer import StockAnalyzer

def main():
    """
    Main function to demonstrate StockAnalyzer capabilities.
    """
    print("Stock Market Performance Analysis Example")
    print("=" * 50)
    
    # Define tickers (Tech giants)
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    print(f"Analyzing stocks: {', '.join(tickers)}")
    
    # Define date range (5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize analyzer
    analyzer = StockAnalyzer(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    # Fetch data
    print("\nFetching historical stock data...")
    data = analyzer.fetch_data()
    
    # Display data sample
    print("\nData sample:")
    print(data.head())
    
    # Calculate returns
    print("\nCalculating daily returns...")
    returns = analyzer.calculate_returns()
    
    # Display returns sample
    print("\nReturns sample:")
    print(returns.head())
    
    # Calculate statistics
    print("\nCalculating key statistics...")
    stats = analyzer.calculate_statistics()
    print("\nStock Statistics:")
    print(stats)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("1. Plotting stock prices...")
    analyzer.plot_prices()
    
    print("2. Plotting return distributions...")
    analyzer.plot_returns_distribution()
    
    print("3. Plotting correlation matrix...")
    analyzer.plot_correlation_matrix()
    
    print("4. Plotting rolling statistics...")
    analyzer.plot_rolling_statistics()
    
    print("5. Plotting cumulative returns...")
    analyzer.plot_cumulative_returns()
    
    print("6. Analyzing and plotting drawdowns...")
    analyzer.plot_drawdowns()
    drawdowns = analyzer.analyze_drawdowns()
    
    # Build prediction models
    print("\nBuilding prediction models...")
    
    for ticker in tickers:
        print(f"Building linear model for {ticker}...")
        analyzer.build_prediction_model(ticker, feature_window=30, target_days=5, model_type='linear')
        analyzer.plot_model_performance(ticker)
        
        # Uncomment to also build ARIMA models (takes longer)
        # print(f"Building ARIMA model for {ticker}...")
        # analyzer.build_prediction_model(ticker, model_type='arima')
        # analyzer.plot_model_performance(ticker)
    
    # Create interactive dashboard
    print("\nCreating interactive dashboard...")
    analyzer.create_interactive_dashboard()
    
    print("\nAnalysis completed! Check the 'outputs' directory for visualizations.")
    print("\nFiles generated:")
    print("- Stock data: data/stock_prices.csv")
    print("- Daily returns: data/daily_returns.csv")
    print("- Stock statistics: data/stock_statistics.csv")
    print("- Static visualizations: outputs/*.png")
    print("- Drawdown analysis: outputs/drawdown_analysis.json")
    print("- Interactive dashboard: outputs/interactive_dashboard.html")

if __name__ == "__main__":
    main()