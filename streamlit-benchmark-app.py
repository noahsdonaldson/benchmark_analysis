import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import glob
from pathlib import Path
import zipfile
import tempfile
from typing import List, Dict, Any
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Benchmark Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > label {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

class BenchmarkProcessor:
    """Backend class to process benchmark JSON files"""
    
    def __init__(self):
        self.combined_data = []
        self.metrics_config = {
            'time_to_first_token': {
                'name': 'Time to First Token',
                'unit': 'ms',
                'color': '#1f77b4'
            },
            'request_latency': {
                'name': 'Request Latency',
                'unit': 'ms',
                'color': '#ff7f0e'
            },
            'output_token_throughput_per_request': {
                'name': 'Output Token Throughput per Request',
                'unit': 'tokens/sec',
                'color': '#2ca02c'
            },
            'output_sequence_length': {
                'name': 'Output Sequence Length',
                'unit': 'tokens',
                'color': '#d62728'
            },
            'input_sequence_length': {
                'name': 'Input Sequence Length',
                'unit': 'tokens',
                'color': '#9467bd'
            },
            'request_throughput': {
                'name': 'Request Throughput',
                'unit': 'requests/sec',
                'color': '#8c564b'
            },
            'output_token_throughput': {
                'name': 'Output Token Throughput',
                'unit': 'tokens/sec',
                'color': '#e377c2'
            }
        }
        self.statistical_fields = ['avg', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'min', 'max', 'std']
    
    def process_uploaded_files(self, uploaded_files) -> bool:
        """Process uploaded files and extract benchmark data"""
        self.combined_data = []
        
        try:
            for uploaded_file in uploaded_files:
                if not (uploaded_file.name.endswith('.json') and 'benchmark' in uploaded_file.name):
                    st.warning(f"Skipping {uploaded_file.name} - doesn't match pattern *benchmark*.json")
                    continue
                
                # Read and parse JSON
                content = uploaded_file.read()
                json_data = json.loads(content)
                
                # Extract run identifier from filename
                run_id = uploaded_file.name.replace('_genai_perf.json', '').replace('benchmark', 'run')
                
                # Create run data structure
                run_data = {
                    'run_id': run_id,
                    'file_name': uploaded_file.name,
                    'request_count': json_data.get('input_config', {}).get('request_count', 0),
                    'model': json_data.get('input_config', {}).get('model', ['unknown'])[0] if json_data.get('input_config', {}).get('model') else 'unknown',
                    'backend': json_data.get('input_config', {}).get('backend', 'unknown'),
                    'concurrency': json_data.get('input_config', {}).get('concurrency', 0),
                    'raw_data': json_data
                }
                
                # Extract metrics
                for metric_key in self.metrics_config.keys():
                    if metric_key in json_data:
                        run_data[metric_key] = json_data[metric_key]
                
                self.combined_data.append(run_data)
            
            return len(self.combined_data) > 0
            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            return False
    
    def process_directory(self, directory_path: str) -> bool:
        """Process all benchmark JSON files in a directory"""
        self.combined_data = []
        
        try:
            pattern = os.path.join(directory_path, "**/*benchmark*.json")
            json_files = glob.glob(pattern, recursive=True)
            
            if not json_files:
                return False
            
            for file_path in json_files:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Extract run identifier from filename
                file_name = os.path.basename(file_path)
                run_id = file_name.replace('_genai_perf.json', '').replace('benchmark', 'run')
                
                # Create run data structure
                run_data = {
                    'run_id': run_id,
                    'file_name': file_name,
                    'file_path': file_path,
                    'request_count': json_data.get('input_config', {}).get('request_count', 0),
                    'model': json_data.get('input_config', {}).get('model', ['unknown'])[0] if json_data.get('input_config', {}).get('model') else 'unknown',
                    'backend': json_data.get('input_config', {}).get('backend', 'unknown'),
                    'concurrency': json_data.get('input_config', {}).get('concurrency', 0),
                    'raw_data': json_data
                }
                
                # Extract metrics
                for metric_key in self.metrics_config.keys():
                    if metric_key in json_data:
                        run_data[metric_key] = json_data[metric_key]
                
                self.combined_data.append(run_data)
            
            return len(self.combined_data) > 0
            
        except Exception as e:
            st.error(f"Error processing directory: {str(e)}")
            return False
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all runs"""
        if not self.combined_data:
            return {}
        
        return {
            'total_runs': len(self.combined_data),
            'unique_models': len(set(run['model'] for run in self.combined_data)),
            'total_requests': sum(run['request_count'] for run in self.combined_data),
            'models_list': list(set(run['model'] for run in self.combined_data)),
            'backends_list': list(set(run['backend'] for run in self.combined_data))
        }
    
    def prepare_chart_data(self, metric: str) -> pd.DataFrame:
        """Prepare data for charting a specific metric"""
        if not self.combined_data:
            return pd.DataFrame()
        
        chart_data = []
        
        for i, run in enumerate(self.combined_data):
            if metric not in run or not run[metric]:
                continue
            
            metric_data = run[metric]
            row = {
                'run_id': run['run_id'],
                'run_index': i + 1,
                'request_count': run['request_count'],
                'model': run['model'],
                'backend': run['backend'],
                'concurrency': run['concurrency'],
                'file_name': run['file_name']
            }
            
            # Add statistical measures
            for field in self.statistical_fields:
                if field in metric_data:
                    row[field] = metric_data[field]
            
            chart_data.append(row)
        
        return pd.DataFrame(chart_data)

def create_metric_charts(processor: BenchmarkProcessor, metric: str):
    """Create charts for a specific metric"""
    data = processor.prepare_chart_data(metric)
    
    if data.empty:
        st.warning(f"No data available for {metric}")
        return
    
    config = processor.metrics_config[metric]
    
    # Create tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend Analysis", "📊 Percentile Distribution", "📉 Min/Max Range", "🔢 Data Table"])
    
    with tab1:
        st.subheader(f"{config['name']} - Average Values Trend")
        
        if 'avg' in data.columns:
            fig = px.line(
                data, 
                x='run_index', 
                y='avg',
                hover_data=['run_id', 'model', 'request_count', 'concurrency'],
                title=f"Average {config['name']} by Run",
                labels={
                    'run_index': 'Run Number',
                    'avg': f"Average ({config['unit']})"
                }
            )
            fig.update_traces(line=dict(color=config['color'], width=3))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Average data not available for this metric")
    
    with tab2:
        st.subheader(f"{config['name']} - Percentile Distribution")
        
        percentile_cols = ['p25', 'p50', 'p75', 'p90', 'p95']
        available_percentiles = [col for col in percentile_cols if col in data.columns]
        
        if available_percentiles:
            fig = go.Figure()
            
            colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']
            for i, col in enumerate(available_percentiles):
                fig.add_trace(go.Scatter(
                    x=data['run_index'],
                    y=data[col],
                    mode='lines+markers',
                    name=col.upper(),
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig.update_layout(
                title=f"Percentile Distribution - {config['name']}",
                xaxis_title="Run Number",
                yaxis_title=f"{config['name']} ({config['unit']})",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Percentile data not available for this metric")
    
    with tab3:
        st.subheader(f"{config['name']} - Min/Max Range")
        
        if all(col in data.columns for col in ['min', 'max', 'avg']):
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data['run_index'],
                y=data['min'],
                mode='lines+markers',
                name='Minimum',
                line=dict(color='#2ca02c', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['run_index'],
                y=data['avg'],
                mode='lines+markers',
                name='Average',
                line=dict(color=config['color'], width=3, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=data['run_index'],
                y=data['max'],
                mode='lines+markers',
                name='Maximum',
                line=dict(color='#d62728', width=2)
            ))
            
            fig.update_layout(
                title=f"Min/Max Range - {config['name']}",
                xaxis_title="Run Number",
                yaxis_title=f"{config['name']} ({config['unit']})",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Min/Max data not available for this metric")
    
    with tab4:
        st.subheader("Raw Data")
        st.dataframe(data, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = BenchmarkProcessor()
    
    # Header
    st.title("📊 Benchmark Data Analyzer")
    st.markdown("Analyze and visualize performance metrics from multiple benchmark JSON files")
    
    # Sidebar for data input
    with st.sidebar:
        st.header("📁 Data Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["Upload Files", "Directory Path"],
            help="Upload individual files or specify a directory path containing benchmark files"
        )
        
        if input_method == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload benchmark JSON files",
                type=['json'],
                accept_multiple_files=True,
                help="Select multiple *benchmark*_genai_perf.json files"
            )
            
            if uploaded_files:
                if st.button("Process Files", type="primary"):
                    with st.spinner("Processing files..."):
                        success = st.session_state.processor.process_uploaded_files(uploaded_files)
                        if success:
                            st.success(f"Successfully processed {len(st.session_state.processor.combined_data)} files!")
                        else:
                            st.error("Failed to process files")
        
        else:
            directory_path = st.text_input(
                "Directory Path",
                placeholder="/path/to/benchmark/files",
                help="Enter the path to directory containing benchmark JSON files"
            )
            
            if directory_path and st.button("Process Directory", type="primary"):
                with st.spinner("Processing directory..."):
                    success = st.session_state.processor.process_directory(directory_path)
                    if success:
                        st.success(f"Successfully processed {len(st.session_state.processor.combined_data)} files!")
                    else:
                        st.error("Failed to process directory or no benchmark files found")
    
    # Main content
    if st.session_state.processor.combined_data:
        # Summary statistics
        st.header("📈 Summary Statistics")
        stats = st.session_state.processor.get_summary_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Runs", stats['total_runs'])
        
        with col2:
            st.metric("Unique Models", stats['unique_models'])
        
        with col3:
            st.metric("Total Requests", f"{stats['total_requests']:,}")
        
        with col4:
            st.metric("Backends", len(stats['backends_list']))
        
        # Models and backends info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Models:** {', '.join(stats['models_list'])}")
        with col2:
            st.info(f"**Backends:** {', '.join(stats['backends_list'])}")
        
        st.divider()
        
        # Metric selection and visualization
        st.header("📊 Metric Analysis")
        
        # Metric selector
        available_metrics = [
            key for key in st.session_state.processor.metrics_config.keys()
            if any(key in run for run in st.session_state.processor.combined_data)
        ]
        
        selected_metric = st.selectbox(
            "Select metric to analyze:",
            available_metrics,
            format_func=lambda x: st.session_state.processor.metrics_config[x]['name'],
            help="Choose a metric to visualize across all runs"
        )
        
        if selected_metric:
            create_metric_charts(st.session_state.processor, selected_metric)
        
        # Raw data export
        st.header("💾 Data Export")
        if st.button("Download Combined Data as JSON"):
            combined_json = json.dumps(st.session_state.processor.combined_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=combined_json,
                file_name="combined_benchmark_data.json",
                mime="application/json"
            )
    
    else:
        st.info("👆 Please upload benchmark JSON files or specify a directory path to get started")
        
        # Show example of expected file structure
        with st.expander("ℹ️ Expected File Format"):
            st.code('''
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
  "input_config": {
    "request_count": 50,
    "model": ["tinyllama"],
    "backend": "tensorrtllm",
    "concurrency": 4
  }
}
            ''', language='json')

if __name__ == "__main__":
    main()
