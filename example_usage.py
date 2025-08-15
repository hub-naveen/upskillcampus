#!/usr/bin/env python3
"""
Example usage of the Smart City Traffic Analysis System

This script demonstrates how to use the TrafficAnalyzer class
for different types of analysis and customizations.
"""

from TrafficManagementSystem import TrafficAnalyzer
import pandas as pd

def main():
    print("🚗 Smart City Traffic Analysis - Example Usage")
    print("=" * 50)
    
    # Initialize the analyzer
    analyzer = TrafficAnalyzer('data/train_aWnotuB.csv', 'data/datasets_8494_11879_test_BdBKkAj.csv')
    
    # Example 1: Basic data exploration
    print("\n📊 Example 1: Basic Data Exploration")
    print("-" * 30)
    analyzer.load_data()
    analyzer.explore_data()
    
    # Example 2: Focus on specific junction analysis
    print("\n🔍 Example 2: Junction-Specific Analysis")
    print("-" * 30)
    
    # Load and process data
    analyzer.load_data()
    analyzer.processed_data = analyzer.add_time_features(analyzer.train_data)
    
    # Analyze specific junction
    junction_1_data = analyzer.processed_data[analyzer.processed_data['Junction'] == 1]
    print(f"Junction 1 Statistics:")
    print(f"Total records: {len(junction_1_data)}")
    print(f"Average vehicles: {junction_1_data['Vehicles'].mean():.2f}")
    print(f"Peak hour: {junction_1_data.groupby('Hour')['Vehicles'].mean().idxmax()}:00")
    
    # Example 3: Custom time period analysis
    print("\n⏰ Example 3: Custom Time Period Analysis")
    print("-" * 30)
    
    # Analyze morning rush hour (7-9 AM)
    morning_data = analyzer.processed_data[
        (analyzer.processed_data['Hour'] >= 7) & 
        (analyzer.processed_data['Hour'] <= 9)
    ]
    
    print(f"Morning Rush Hour (7-9 AM) Analysis:")
    print(f"Average vehicles: {morning_data['Vehicles'].mean():.2f}")
    print(f"Busiest junction: {morning_data.groupby('Junction')['Vehicles'].mean().idxmax()}")
    
    # Example 4: Weekend analysis
    print("\n🏖️ Example 4: Weekend Traffic Analysis")
    print("-" * 30)
    
    weekend_data = analyzer.processed_data[analyzer.processed_data['IsWeekend'] == 1]
    weekday_data = analyzer.processed_data[analyzer.processed_data['IsWeekend'] == 0]
    
    print(f"Weekend average: {weekend_data['Vehicles'].mean():.2f}")
    print(f"Weekday average: {weekday_data['Vehicles'].mean():.2f}")
    print(f"Weekend/Weekday ratio: {weekend_data['Vehicles'].mean() / weekday_data['Vehicles'].mean():.2f}")
    
    # Example 5: Seasonal analysis
    print("\n🌤️ Example 5: Seasonal Traffic Patterns")
    print("-" * 30)
    
    seasonal_stats = analyzer.processed_data.groupby('Season')['Vehicles'].agg(['mean', 'std'])
    print("Seasonal Traffic Statistics:")
    print(seasonal_stats.round(2))
    
    # Example 6: Peak traffic identification
    print("\n📈 Example 6: Peak Traffic Identification")
    print("-" * 30)
    
    hourly_avg = analyzer.processed_data.groupby('Hour')['Vehicles'].mean()
    peak_hours = hourly_avg.nlargest(3)
    
    print("Top 3 Peak Hours:")
    for hour, avg_vehicles in peak_hours.items():
        print(f"  {hour}:00 - {avg_vehicles:.2f} vehicles")
    
    # Example 7: Generate custom recommendations
    print("\n💡 Example 7: Custom Recommendations")
    print("-" * 30)
    
    # Analyze each junction for specific recommendations
    for junction in range(1, 5):
        junction_data = analyzer.processed_data[analyzer.processed_data['Junction'] == junction]
        avg_traffic = junction_data['Vehicles'].mean()
        peak_hour = junction_data.groupby('Hour')['Vehicles'].mean().idxmax()
        
        print(f"\nJunction {junction}:")
        print(f"  Average traffic: {avg_traffic:.1f} vehicles")
        print(f"  Peak hour: {peak_hour}:00")
        
        if avg_traffic > 25:
            print("  Priority: HIGH - Consider smart traffic signals and lane expansion")
        elif avg_traffic > 15:
            print("  Priority: MEDIUM - Monitor and optimize existing infrastructure")
        else:
            print("  Priority: LOW - Standard traffic management sufficient")
    
    print("\n✅ Example analysis complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
