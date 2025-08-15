# 📊 Data Folder

This folder contains the traffic datasets used for the Smart City Traffic Analysis project.

## 📁 Files

### `train_aWnotuB.csv` (1.7MB)
- **Purpose**: Training dataset with historical traffic data
- **Records**: 48,120 traffic measurements
- **Time Period**: November 2015 to June 2017
- **Columns**:
  - `DateTime`: Timestamp of measurement
  - `Junction`: Traffic junction ID (1-4)
  - `Vehicles`: Number of vehicles counted
  - `ID`: Unique record identifier

### `datasets_8494_11879_test_BdBKkAj.csv` (404KB)
- **Purpose**: Test dataset for future traffic predictions
- **Records**: 11,808 future timestamps
- **Time Period**: Future dates for forecasting
- **Columns**:
  - `DateTime`: Future timestamp for prediction
  - `Junction`: Traffic junction ID (1-4)
  - `ID`: Unique record identifier

## 🎯 Data Quality
- ✅ No missing values
- ✅ Consistent data format
- ✅ 4 junctions covered
- ✅ Hourly measurements
- ✅ 20-month historical data

## 📈 Data Insights
- **Peak Hour**: 19:00 (7 PM) - 29.85 vehicles
- **Busiest Junction**: Junction 1 - 45.05 vehicles/hour
- **Weekend Traffic**: 28% less than weekdays
- **Total Period**: 607 days of data

## 🔧 Usage
All Python scripts automatically reference these files from the `data/` folder:
- `traffic_analysis.py` - Main analysis system
- `quick_analysis.py` - Fast insights
- `example_usage.py` - Learning examples
- `Intern.ipynb` - Learning notebook
