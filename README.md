# Benchmark Data Analyzer - Setup Instructions

## Quick Setup

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### Option 1: Upload Files
1. Click "Upload Files" in the sidebar
2. Select multiple `*benchmark*_genai_perf.json` files
3. Click "Process Files"

### Option 2: Directory Path
1. Select "Directory Path" in the sidebar
2. Enter the path to your directory containing benchmark files
3. Click "Process Directory"

## Features

### Backend Processing
- **File Processing**: Automatically identifies and processes benchmark JSON files
- **Data Validation**: Checks file format and structure
- **Flexible Input**: Supports file upload or directory scanning
- **Error Handling**: Graceful handling of malformed files

### Frontend Visualization
- **Summary Dashboard**: Overview of all runs, models, and requests
- **Multiple Chart Types**: 
  - Trend analysis (average values over time)
  - Percentile distribution charts
  - Min/Max range visualization
  - Raw data tables
- **Interactive Charts**: Hover tooltips with detailed information
- **Metric Selection**: Choose from available metrics
- **Data Export**: Download combined data as JSON

### Supported Metrics
- Time to First Token
- Request Latency  
- Output Token Throughput per Request
- Output/Input Sequence Length
- Request Throughput
- Output Token Throughput

## File Structure

```
benchmark-analyzer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Architecture

### Backend (`BenchmarkProcessor` class)
- Handles file processing and data extraction
- Validates JSON structure and extracts metrics
- Combines data from multiple runs
- Prepares data for visualization

### Frontend (Streamlit UI)
- Interactive file upload/directory selection
- Real-time data processing feedback
- Multiple visualization tabs
- Summary statistics dashboard
- Data export functionality

## Troubleshooting

### Common Issues

1. **Files not being processed**: Ensure filenames contain "benchmark" and end with ".json"
2. **Missing metrics**: Check that your JSON files contain the expected metric structure
3. **Installation errors**: Make sure you're using Python 3.8+ and have activated your virtual environment

### Performance Considerations
- Large numbers of files (100+) may take longer to process
- The app processes files sequentially for reliability
- Charts are rendered client-side using Plotly for smooth interaction

## Customization

You can easily extend the application by:
- Adding new metrics to the `metrics_config` dictionary
- Modifying chart types and styling
- Adding additional statistical analysis
- Implementing data filtering and search capabilities
