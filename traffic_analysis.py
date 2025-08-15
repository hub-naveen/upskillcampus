import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrafficAnalyzer:
    def __init__(self, train_file='data/train_aWnotuB.csv', test_file='data/datasets_8494_11879_test_BdBKkAj.csv'):
        """
        Initialize the Traffic Analyzer for smart city traffic management
        
        Args:
            train_file (str): Path to training dataset (default: data/train_aWnotuB.csv)
            test_file (str): Path to test dataset (default: data/datasets_8494_11879_test_BdBKkAj.csv)
        """
        self.train_file = train_file
        self.test_file = test_file
        self.train_data = None
        self.test_data = None
        self.processed_data = None
        
    def load_data(self):
        """Load and preprocess the traffic data"""
        print("Loading traffic data...")
        
        # Load training data
        self.train_data = pd.read_csv(self.train_file)
        self.train_data['DateTime'] = pd.to_datetime(self.train_data['DateTime'])
        
        # Load test data
        self.test_data = pd.read_csv(self.test_file)
        self.test_data['DateTime'] = pd.to_datetime(self.test_data['DateTime'])
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        # Basic data info
        print("\nTraining Data Info:")
        print(self.train_data.info())
        print("\nTest Data Info:")
        print(self.test_data.info())
        
        return self
    
    def explore_data(self):
        """Perform comprehensive data exploration"""
        print("\n" + "="*50)
        print("TRAFFIC DATA EXPLORATION")
        print("="*50)
        
        # Data range
        print(f"\nData Time Range:")
        print(f"Training: {self.train_data['DateTime'].min()} to {self.train_data['DateTime'].max()}")
        print(f"Test: {self.test_data['DateTime'].min()} to {self.test_data['DateTime'].max()}")
        
        # Junction distribution
        print(f"\nJunction Distribution:")
        print(self.train_data['Junction'].value_counts().sort_index())
        
        # Vehicle statistics
        print(f"\nVehicle Count Statistics:")
        print(self.train_data['Vehicles'].describe())
        
        # Check for missing values
        print(f"\nMissing Values:")
        print(self.train_data.isnull().sum())
        
        return self
    
    def add_time_features(self, df):
        """Add time-based features for analysis"""
        df = df.copy()
        
        # Extract time components
        df['Year'] = df['DateTime'].dt.year
        df['Month'] = df['DateTime'].dt.month
        df['Day'] = df['DateTime'].dt.day
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['DayOfYear'] = df['DateTime'].dt.dayofyear
        df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week
        
        # Create time periods
        df['TimePeriod'] = pd.cut(df['Hour'], 
                                 bins=[0, 6, 12, 18, 24], 
                                 labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Weekend vs Weekday
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Season
        df['Season'] = pd.cut(df['Month'], 
                             bins=[0, 3, 6, 9, 12], 
                             labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        return df
    
    def analyze_traffic_patterns(self):
        """Analyze traffic patterns across different dimensions"""
        print("\n" + "="*50)
        print("TRAFFIC PATTERN ANALYSIS")
        print("="*50)
        
        # Add time features
        self.processed_data = self.add_time_features(self.train_data)
        
        # Create comprehensive analysis
        self._analyze_hourly_patterns()
        self._analyze_daily_patterns()
        self._analyze_weekly_patterns()
        self._analyze_monthly_patterns()
        self._analyze_junction_comparison()
        self._analyze_weekend_vs_weekday()
        self._analyze_seasonal_patterns()
        
        return self
    
    def _analyze_hourly_patterns(self):
        """Analyze hourly traffic patterns"""
        print("\n1. Hourly Traffic Patterns")
        
        hourly_avg = self.processed_data.groupby('Hour')['Vehicles'].mean()
        
        plt.figure(figsize=(12, 6))
        hourly_avg.plot(kind='bar', color='skyblue', alpha=0.7)
        plt.title('Average Traffic by Hour of Day', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Average Number of Vehicles', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('hourly_traffic_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Peak hours identification
        peak_hours = hourly_avg.nlargest(5)
        print(f"Peak Hours: {peak_hours.index.tolist()}")
        print(f"Peak Hour Averages: {peak_hours.values.round(2)}")
        
    def _analyze_daily_patterns(self):
        """Analyze daily traffic patterns"""
        print("\n2. Daily Traffic Patterns")
        
        daily_avg = self.processed_data.groupby('DayOfWeek')['Vehicles'].mean()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        plt.figure(figsize=(12, 6))
        daily_avg.plot(kind='bar', color='lightcoral', alpha=0.7)
        plt.title('Average Traffic by Day of Week', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('Average Number of Vehicles', fontsize=12)
        plt.xticks(range(7), day_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('daily_traffic_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _analyze_weekly_patterns(self):
        """Analyze weekly traffic patterns"""
        print("\n3. Weekly Traffic Patterns")
        
        weekly_avg = self.processed_data.groupby('WeekOfYear')['Vehicles'].mean()
        
        plt.figure(figsize=(15, 6))
        weekly_avg.plot(kind='line', marker='o', color='green', alpha=0.7)
        plt.title('Average Traffic by Week of Year', fontsize=14, fontweight='bold')
        plt.xlabel('Week of Year', fontsize=12)
        plt.ylabel('Average Number of Vehicles', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('weekly_traffic_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _analyze_monthly_patterns(self):
        """Analyze monthly traffic patterns"""
        print("\n4. Monthly Traffic Patterns")
        
        monthly_avg = self.processed_data.groupby('Month')['Vehicles'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        plt.figure(figsize=(12, 6))
        monthly_avg.plot(kind='bar', color='orange', alpha=0.7)
        plt.title('Average Traffic by Month', fontsize=14, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Average Number of Vehicles', fontsize=12)
        plt.xticks(range(12), month_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('monthly_traffic_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _analyze_junction_comparison(self):
        """Compare traffic patterns across junctions"""
        print("\n5. Junction Comparison")
        
        junction_hourly = self.processed_data.groupby(['Junction', 'Hour'])['Vehicles'].mean().unstack()
        
        plt.figure(figsize=(15, 8))
        for junction in junction_hourly.index:
            plt.plot(junction_hourly.columns, junction_hourly.loc[junction], 
                    marker='o', label=f'Junction {junction}', linewidth=2)
        
        plt.title('Hourly Traffic Patterns by Junction', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Average Number of Vehicles', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('junction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Junction statistics
        junction_stats = self.processed_data.groupby('Junction')['Vehicles'].agg(['mean', 'std', 'min', 'max'])
        print("\nJunction Statistics:")
        print(junction_stats.round(2))
        
    def _analyze_weekend_vs_weekday(self):
        """Analyze weekend vs weekday patterns"""
        print("\n6. Weekend vs Weekday Analysis")
        
        weekend_weekday = self.processed_data.groupby(['IsWeekend', 'Hour'])['Vehicles'].mean().unstack()
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        weekend_weekday.loc[0].plot(kind='bar', color='blue', alpha=0.7, title='Weekday Traffic')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Vehicles')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        weekend_weekday.loc[1].plot(kind='bar', color='red', alpha=0.7, title='Weekend Traffic')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Vehicles')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('weekend_weekday_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _analyze_seasonal_patterns(self):
        """Analyze seasonal traffic patterns"""
        print("\n7. Seasonal Analysis")
        
        seasonal_hourly = self.processed_data.groupby(['Season', 'Hour'])['Vehicles'].mean().unstack()
        
        plt.figure(figsize=(15, 8))
        for season in seasonal_hourly.index:
            plt.plot(seasonal_hourly.columns, seasonal_hourly.loc[season], 
                    marker='o', label=season, linewidth=2)
        
        plt.title('Hourly Traffic Patterns by Season', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Average Number of Vehicles', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def identify_holiday_patterns(self):
        """Identify potential holiday patterns"""
        print("\n" + "="*50)
        print("HOLIDAY PATTERN ANALYSIS")
        print("="*50)
        
        # Create holiday indicators (simplified approach)
        self.processed_data['IsHoliday'] = 0
        
        # Common holiday periods (simplified)
        holiday_periods = [
            # New Year
            ((1, 1), (1, 2)),
            # Republic Day (India)
            ((1, 26), (1, 26)),
            # Independence Day (India)
            ((8, 15), (8, 15)),
            # Diwali (approximate)
            ((10, 20), (11, 10)),
            # Christmas
            ((12, 25), (12, 25)),
        ]
        
        for start, end in holiday_periods:
            mask = ((self.processed_data['Month'] == start[0]) & 
                   (self.processed_data['Day'] >= start[1])) | \
                  ((self.processed_data['Month'] == end[0]) & 
                   (self.processed_data['Day'] <= end[1]))
            self.processed_data.loc[mask, 'IsHoliday'] = 1
        
        # Compare holiday vs non-holiday traffic
        holiday_comparison = self.processed_data.groupby('IsHoliday')['Vehicles'].agg(['mean', 'std'])
        print("\nHoliday vs Non-Holiday Traffic:")
        print(holiday_comparison.round(2))
        
        # Holiday hourly patterns
        holiday_hourly = self.processed_data.groupby(['IsHoliday', 'Hour'])['Vehicles'].mean().unstack()
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        holiday_hourly.loc[0].plot(kind='bar', color='blue', alpha=0.7, title='Regular Day Traffic')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Vehicles')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        holiday_hourly.loc[1].plot(kind='bar', color='red', alpha=0.7, title='Holiday Traffic')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Vehicles')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('holiday_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def generate_forecasting_model(self):
        """Generate time series forecasting model"""
        print("\n" + "="*50)
        print("TRAFFIC FORECASTING MODEL")
        print("="*50)
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Prepare features for forecasting
            forecast_data = self.processed_data.copy()
            
            # Create lag features
            for lag in [1, 2, 3, 6, 12, 24]:
                forecast_data[f'lag_{lag}'] = forecast_data.groupby('Junction')['Vehicles'].shift(lag)
            
            # Create rolling features
            for window in [3, 6, 12, 24]:
                forecast_data[f'rolling_mean_{window}'] = forecast_data.groupby('Junction')['Vehicles'].rolling(window).mean().reset_index(0, drop=True)
                forecast_data[f'rolling_std_{window}'] = forecast_data.groupby('Junction')['Vehicles'].rolling(window).std().reset_index(0, drop=True)
            
            # Drop NaN values
            forecast_data = forecast_data.dropna()
            
            # Select features
            feature_columns = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsHoliday'] + \
                            [col for col in forecast_data.columns if col.startswith('lag_') or col.startswith('rolling_')]
            
            X = forecast_data[feature_columns]
            y = forecast_data['Vehicles']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate model
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nModel Performance:")
            print(f"Mean Absolute Error: {mae:.2f}")
            print(f"Root Mean Square Error: {rmse:.2f}")
            print(f"R² Score: {r2:.3f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Plot predictions vs actual
            plt.figure(figsize=(12, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Traffic')
            plt.ylabel('Predicted Traffic')
            plt.title('Traffic Prediction: Actual vs Predicted')
            plt.tight_layout()
            plt.savefig('prediction_accuracy.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return model, feature_importance
            
        except ImportError:
            print("Scikit-learn not available. Skipping forecasting model.")
            return None, None
    
    def generate_recommendations(self):
        """Generate infrastructure and traffic management recommendations"""
        print("\n" + "="*50)
        print("SMART CITY TRAFFIC RECOMMENDATIONS")
        print("="*50)
        
        # Analyze peak traffic periods
        hourly_avg = self.processed_data.groupby('Hour')['Vehicles'].mean()
        peak_hours = hourly_avg.nlargest(5)
        
        # Analyze junction-specific patterns
        junction_peaks = self.processed_data.groupby(['Junction', 'Hour'])['Vehicles'].mean().unstack()
        
        # Weekend vs weekday analysis
        weekend_weekday = self.processed_data.groupby(['IsWeekend', 'Hour'])['Vehicles'].mean().unstack()
        
        print("\n1. TRAFFIC PATTERN INSIGHTS:")
        print(f"   • Peak traffic hours: {peak_hours.index.tolist()}")
        print(f"   • Peak hour averages: {peak_hours.values.round(2)}")
        
        # Junction-specific recommendations
        print("\n2. JUNCTION-SPECIFIC RECOMMENDATIONS:")
        for junction in range(1, 5):
            junction_data = self.processed_data[self.processed_data['Junction'] == junction]
            avg_traffic = junction_data['Vehicles'].mean()
            peak_hour = junction_data.groupby('Hour')['Vehicles'].mean().idxmax()
            
            print(f"   Junction {junction}:")
            print(f"     - Average daily traffic: {avg_traffic:.1f} vehicles")
            print(f"     - Peak hour: {peak_hour}:00")
            print(f"     - Recommendation: {'High priority' if avg_traffic > 25 else 'Medium priority' if avg_traffic > 15 else 'Low priority'} for infrastructure upgrades")
        
        print("\n3. INFRASTRUCTURE PLANNING RECOMMENDATIONS:")
        print("   • Implement smart traffic signals with AI-based timing optimization")
        print("   • Deploy real-time traffic monitoring systems at all junctions")
        print("   • Install dynamic message signs for traffic alerts")
        print("   • Consider expanding lanes during peak hours at high-traffic junctions")
        print("   • Implement smart parking systems to reduce congestion")
        
        print("\n4. OPERATIONAL RECOMMENDATIONS:")
        print("   • Increase traffic police presence during peak hours")
        print("   • Implement flexible working hours to reduce rush hour congestion")
        print("   • Develop mobile apps for real-time traffic updates")
        print("   • Establish emergency response corridors")
        
        print("\n5. LONG-TERM STRATEGIC RECOMMENDATIONS:")
        print("   • Develop comprehensive public transportation network")
        print("   • Implement bike-sharing and pedestrian-friendly infrastructure")
        print("   • Consider congestion pricing during peak hours")
        print("   • Invest in electric vehicle charging infrastructure")
        print("   • Develop smart city command center for integrated traffic management")
        
        return self
    
    def create_dashboard_data(self):
        """Create data for real-time dashboard"""
        print("\n" + "="*50)
        print("DASHBOARD DATA PREPARATION")
        print("="*50)
        
        # Create summary statistics for dashboard
        dashboard_data = {
            'total_records': len(self.processed_data),
            'date_range': {
                'start': self.processed_data['DateTime'].min().strftime('%Y-%m-%d'),
                'end': self.processed_data['DateTime'].max().strftime('%Y-%m-%d')
            },
            'junctions': self.processed_data['Junction'].nunique(),
            'avg_daily_traffic': self.processed_data['Vehicles'].mean(),
            'peak_hour': self.processed_data.groupby('Hour')['Vehicles'].mean().idxmax(),
            'busiest_junction': self.processed_data.groupby('Junction')['Vehicles'].mean().idxmax(),
            'weekend_traffic_ratio': self.processed_data[self.processed_data['IsWeekend'] == 1]['Vehicles'].mean() / 
                                   self.processed_data[self.processed_data['IsWeekend'] == 0]['Vehicles'].mean()
        }
        
        print("Dashboard Summary Statistics:")
        for key, value in dashboard_data.items():
            print(f"   {key}: {value}")
        
        return dashboard_data
    
    def run_complete_analysis(self):
        """Run the complete traffic analysis pipeline"""
        print("🚗 SMART CITY TRAFFIC ANALYSIS SYSTEM 🚗")
        print("="*60)
        
        # Execute analysis pipeline
        (self.load_data()
         .explore_data()
         .analyze_traffic_patterns()
         .identify_holiday_patterns()
         .generate_forecasting_model()
         .generate_recommendations()
         .create_dashboard_data())
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE! Check the generated visualizations and recommendations.")
        print("="*60)
        
        return self

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TrafficAnalyzer('train_aWnotuB.csv', 'datasets_8494_11879_test_BdBKkAj.csv')
    
    # Run complete analysis
    analyzer.run_complete_analysis()
