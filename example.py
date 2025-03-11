#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating the use of StockAnalyzer class with S&P 500 Kaggle dataset.
"""

from stock_analyzer import StockAnalyzer
import os

def main():
    """
    Main function to demonstrate StockAnalyzer capabilities with S&P 500 data.
    """
    print("StockPulse: S&P 500 Market Performance Analysis")
    print("=" * 50)
    
    # Set the path to your Kaggle dataset
    # You should download the dataset from: https://www.kaggle.com/datasets/camnugent/sandp500
    # and extract it to the 'data/kaggle' directory
    data_path = 'data/kaggle'
    
    # Check if the dataset exists
    if not os.path.exists(os.path.join(data_path, 'all_stocks_5yr.csv')):
        print("Error: S&P 500 dataset not found.")
        print("Please download it from: https://www.kaggle.com/datasets/camnugent/sandp500")
        print("and extract it to the 'data/kaggle' directory.")
        return
    
    # Initialize analyzer for technology sector stocks
    print("\nAnalyzing technology sector stocks...")
    tech_analyzer = StockAnalyzer(sector='Information Technology', top_n=10)
    
    # Load data
    tech_data = tech_analyzer.load_kaggle_data(data_path)
    
    # Display data sample
    print("\nTech stocks data sample:")
    print(tech_data.head())
    
    # Calculate returns
    print("\nCalculating daily returns...")
    tech_returns = tech_analyzer.calculate_returns()
    
    # Calculate statistics
    print("\nCalculating key statistics...")
    tech_stats = tech_analyzer.calculate_statistics()
    print("\nTech Stocks Statistics:")
    print(tech_stats)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("1. Plotting tech stock prices...")
    tech_analyzer.plot_prices()
    
    print("2. Plotting tech stock return distributions...")
    tech_analyzer.plot_returns_distribution()
    
    print("3. Plotting correlation matrix...")
    tech_analyzer.plot_correlation_matrix()
    
    print("4. Plotting rolling statistics...")
    tech_analyzer.plot_rolling_statistics()
    
    print("5. Plotting cumulative returns...")
    tech_analyzer.plot_cumulative_returns()
    
    # Analyze top performers across all sectors
    print("\nAnalyzing top performers across all sectors...")
    top_analyzer = StockAnalyzer(top_n=20)
    top_data = top_analyzer.load_kaggle_data(data_path)
    top_returns = top_analyzer.calculate_returns()
    
    print("6. Plotting sector performance...")
    top_analyzer.plot_sector_performance()
    
    print("7. Analyzing and plotting drawdowns...")
    top_tickers = list(top_analyzer.data.columns[:3])  # Top 3 stocks
    drawdowns = top_analyzer.analyze_drawdowns(tickers=top_tickers)
    top_analyzer.plot_drawdowns(tickers=top_tickers)
    
    # Build prediction models
    print("\nBuilding prediction models...")
    top_performer = top_analyzer.data.columns[0]  # First top performer
    
    print(f"Building linear model for {top_performer}...")
    top_analyzer.build_prediction_model(top_performer, feature_window=30, target_days=5, model_type='linear')
    top_analyzer.plot_model_performance(top_performer)
    
    # Create interactive dashboard
    print("\nCreating interactive dashboard...")
    dashboard_tickers = list(top_analyzer.data.columns[:5])  # Top 5 stocks
    top_analyzer.create_interactive_dashboard(tickers=dashboard_tickers)
    
    print("\nAnalysis completed! Check the 'outputs' directory for visualizations.")
    print("\nFiles generated:")
    print("- Stock data: data/sp500_prices.csv")
    print("- Daily returns: data/sp500_daily_returns.csv")
    print("- Stock statistics: data/sp500_statistics.csv")
    print("- Static visualizations: outputs/sp500_*.png")
    print("- Drawdown analysis: outputs/sp500_drawdown_analysis.json")
    print("- Interactive dashboard: outputs/sp500_interactive_dashboard.html")

if __name__ == "__main__":
    main()