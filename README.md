# Advanced Benchmark Data Analyzer

A powerful Streamlit application for analyzing and comparing GenAI performance benchmarks across multiple datasets, models, and platforms. Perfect for evaluating different LLM providers, comparing model performance, and making data-driven decisions about AI infrastructure.

## 🚀 Key Features

### Multi-Dataset Management
- **Label & Organize**: Add meaningful labels to benchmark datasets (e.g., "GPT-4 Azure", "Claude AWS")
- **Bulk Upload**: Process entire folders of benchmark files as labeled datasets
- **Dataset Overview**: View summaries with run counts, models, and performance stats
- **Easy Management**: Add, analyze, or delete datasets through an intuitive interface

### Three Analysis Modes
1. **📂 Dataset Management**: Upload, label, and organize multiple benchmark datasets
2. **🔍 Single Dataset Analysis**: Deep-dive analysis of individual datasets with trend analysis, percentile distributions, and detailed metrics
3. **⚖️ Multi-Dataset Comparison**: Side-by-side performance comparison across 2+ datasets

### Advanced Comparison Features
- **Average Performance Comparison**: Bar charts showing which dataset performs best
- **Range Analysis**: Min/max visualization to understand performance variability
- **Statistical Summaries**: Detailed comparison tables with averages, std dev, and run counts
- **Best Performance Highlighting**: Automatically identifies top-performing datasets
- **Common Metrics**: Smart filtering to show only metrics available across all selected datasets

### Comprehensive Visualizations
- **Trend Analysis**: Performance trends across multiple runs
- **Percentile Distributions**: P25, P50, P75, P90, P95 analysis
- **Min/Max Range Charts**: Understanding performance bounds
- **Interactive Charts**: Hover tooltips with detailed run information
- **Color-Coded Datasets**: Easy visual distinction between different datasets

## 📊 Supported Metrics

- **Time to First Token** (ms)
- **Request Latency** (ms)
- **Output Token Throughput per Request** (tokens/sec)
- **Output/Input Sequence Length** (tokens)
- **Request Throughput** (requests/sec)
- **Output Token Throughput** (tokens/sec)

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone or download the project files
# Navigate to the project directory

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 How to Use

### 1. Adding Datasets

#### Option A: Upload Files
1. Go to "📂 Manage Datasets" mode
2. Enter a descriptive dataset label (e.g., "GPT-4 Production Load")
3. Select "Upload Files" and choose multiple `*benchmark*_genai_perf.json` files
4. Click "Add Dataset"

#### Option B: Directory Upload
1. Go to "📂 Manage Datasets" mode
2. Enter a descriptive dataset label
3. Select "Directory Path" and enter the full path to your benchmark files
4. Click "Add Dataset"

### 2. Single Dataset Analysis
1. Switch to "🔍 Single Dataset Analysis" mode
2. Select a dataset from the dropdown
3. Choose a metric to analyze
4. Explore the four analysis tabs:
   - **Trend Analysis**: See performance over multiple runs
   - **Percentile Distribution**: Understand performance distribution
   - **Min/Max Range**: View performance bounds
   - **Data Table**: Raw data inspection

### 3. Multi-Dataset Comparison
1. Switch to "⚖️ Compare Datasets" mode
2. Select 2 or more datasets to compare
3. Choose a metric available across all selected datasets
4. Review comparison charts:
   - **Average Comparison**: Bar chart of dataset averages
   - **Range Comparison**: Min/max performance ranges
   - **Summary Table**: Detailed statistical comparison

### 4. Export Options
- **Download All Datasets**: Export all labeled datasets as JSON
- **Download Comparison Report**: Export comparison analysis results

## 🎯 Use Cases

### Platform Comparison
Compare the same model across different cloud providers:
- "GPT-4 on Azure OpenAI"
- "GPT-4 on OpenAI Direct"  
- "GPT-4 on AWS Bedrock"

### Model Evaluation
Evaluate different models on the same platform:
- "GPT-4 Turbo"
- "Claude-3 Sonnet"
- "Gemini Pro"
- "LLaMA-2 70B"

### Configuration Testing
Test different deployment configurations:
- "High Concurrency Setup"
- "Standard Load Configuration"
- "Memory Optimized Instance"

### Provider Evaluation
Compare multiple providers for procurement decisions:
- "Azure OpenAI Service"
- "AWS Bedrock"
- "Google Vertex AI"
- "Local Deployment"

## 📁 Expected File Format

Benchmark files should:
- Be named with pattern `*benchmark*_genai_perf.json`
- Follow the GenAI Perf output format

Example structure:
```json
{
  "time_to_first_token": {
    "unit": "ms",
    "avg": 11878.79,
    "p25": 10373.02,
    "p50": 12091.21,
    "p75": 13422.45,
    "p90": 14829.60,
    "p95": 15398.28,
    "p99": 16691.85,
    "min": 5043.75,
    "max": 16816.08,
    "std": 2492.40
  },
  "request_latency": {
    "unit": "ms",
    "avg": 15234.56,
    // ... similar structure
  },
  "input_config": {
    "request_count": 50,
    "model": ["tinyllama"],
    "backend": "tensorrtllm",
    "concurrency": 4
  }
}
```

## 🏗️ Architecture

### Backend (`BenchmarkProcessor` class)
- **Multi-dataset management**: Store and organize labeled datasets
- **Data processing**: Parse GenAI Perf JSON files and extract metrics
- **Comparison engine**: Calculate cross-dataset statistics and comparisons
- **Export functionality**: Generate reports and data exports

### Frontend (Streamlit UI)
- **Three-mode interface**: Dataset management, single analysis, and comparison
- **Interactive visualizations**: Plotly charts with hover details and zooming
- **Real-time processing**: Live feedback during file processing
- **Responsive design**: Works on desktop and tablet devices

## 🔧 Customization

### Adding New Metrics
Edit the `metrics_config` dictionary in `BenchmarkProcessor.__init__()`:

```python
'new_metric_name': {
    'name': 'Display Name',
    'unit': 'unit_type',
    'color': '#hex_color'
}
```

### Modifying Charts
Chart creation functions are modular and can be customized:
- `create_metric_charts()`: Single dataset visualizations
- `create_comparison_charts()`: Multi-dataset comparisons

### Styling
Update the CSS in the `st.markdown()` sections for custom styling.

## 🐛 Troubleshooting

### Common Issues

**Files not being processed**
- Ensure filenames contain "benchmark" and end with ".json"
- Check that JSON structure matches expected GenAI Perf format

**Missing metrics in comparison**
- Comparison only shows metrics available in ALL selected datasets
- Ensure all datasets contain the metrics you want to compare

**Performance with large datasets**
- The app handles 100+ files efficiently
- Processing is done sequentially for data integrity
- Large comparisons (10+ datasets) may take a few seconds to render

**Memory usage**
- Each dataset is stored in memory during the session
- Restart the app if you're working with very large datasets (1000+ runs)

### Performance Tips
- Use descriptive dataset labels for easier identification
- Delete unused datasets to keep the interface clean
- Export important comparisons before adding many new datasets

## 📋 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
numpy>=1.24.0
```

## 🤝 Contributing

This tool is designed to be easily extensible. Common enhancement areas:
- Additional statistical analyses
- New visualization types
- Export formats (PDF, Excel)
- Advanced filtering options
- Automated report generation

## 📄 License

This project is provided as-is for benchmark analysis purposes.