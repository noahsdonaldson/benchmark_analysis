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
from typing import List, Dict, Any, Optional
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
    .dataset-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .comparison-highlight {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 3px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class BenchmarkProcessor:
    """Backend class to process benchmark JSON files"""
    
    def __init__(self):
        self.datasets = {}  # Dictionary to store multiple labeled datasets
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
        
        # Color palette for different datasets
        self.dataset_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def add_dataset(self, dataset_label: str, data: List[Dict]) -> bool:
        """Add a new labeled dataset"""
        if not dataset_label or not data:
            return False
        
        self.datasets[dataset_label] = {
            'data': data,
            'created_at': pd.Timestamp.now(),
            'summary': self._calculate_dataset_summary(data)
        }
        return True
    
    def remove_dataset(self, dataset_label: str) -> bool:
        """Remove a dataset"""
        if dataset_label in self.datasets:
            del self.datasets[dataset_label]
            return True
        return False
    
    def get_dataset_labels(self) -> List[str]:
        """Get list of all dataset labels"""
        return list(self.datasets.keys())
    
    def get_dataset(self, dataset_label: str) -> Optional[Dict]:
        """Get a specific dataset"""
        return self.datasets.get(dataset_label)
    
    def process_uploaded_files(self, uploaded_files, dataset_label: str = None) -> List[Dict]:
        """Process uploaded files and return benchmark data"""
        combined_data = []
        
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
                    'dataset_label': dataset_label,
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
                
                combined_data.append(run_data)
            
            return combined_data
            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            return []
    
    def process_directory(self, directory_path: str, dataset_label: str = None) -> List[Dict]:
        """Process all benchmark JSON files in a directory"""
        combined_data = []
        
        try:
            pattern = os.path.join(directory_path, "**/*benchmark*.json")
            json_files = glob.glob(pattern, recursive=True)
            
            if not json_files:
                return []
            
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
                    'dataset_label': dataset_label,
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
                
                combined_data.append(run_data)
            
            return combined_data
            
        except Exception as e:
            st.error(f"Error processing directory: {str(e)}")
            return []
    
    def _calculate_dataset_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for a dataset"""
        if not data:
            return {}
        
        return {
            'total_runs': len(data),
            'unique_models': len(set(run['model'] for run in data)),
            'total_requests': sum(run['request_count'] for run in data),
            'models_list': list(set(run['model'] for run in data)),
            'backends_list': list(set(run['backend'] for run in data)),
            'avg_concurrency': np.mean([run['concurrency'] for run in data])
        }
    
    def get_dataset_summary(self, dataset_label: str) -> Dict[str, Any]:
        """Get summary statistics for a specific dataset"""
        if dataset_label not in self.datasets:
            return {}
        return self.datasets[dataset_label]['summary']
    
    def prepare_chart_data(self, dataset_label: str, metric: str) -> pd.DataFrame:
        """Prepare data for charting a specific metric from a dataset"""
        if dataset_label not in self.datasets:
            return pd.DataFrame()
        
        data = self.datasets[dataset_label]['data']
        chart_data = []
        
        for i, run in enumerate(data):
            if metric not in run or not run[metric]:
                continue
            
            metric_data = run[metric]
            row = {
                'run_id': run['run_id'],
                'run_index': i + 1,
                'dataset_label': dataset_label,
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
    
    def prepare_comparison_data(self, dataset_labels: List[str], metric: str) -> pd.DataFrame:
        """Prepare data for comparing multiple datasets"""
        if not dataset_labels:
            return pd.DataFrame()
        
        comparison_data = []
        
        for dataset_label in dataset_labels:
            if dataset_label not in self.datasets:
                continue
            
            data = self.datasets[dataset_label]['data']
            
            # Calculate averages for each metric across all runs in the dataset
            metric_values = []
            for run in data:
                if metric in run and run[metric] and 'avg' in run[metric]:
                    metric_values.append(run[metric]['avg'])
            
            if metric_values:
                comparison_data.append({
                    'dataset_label': dataset_label,
                    'avg_value': np.mean(metric_values),
                    'min_value': np.min(metric_values),
                    'max_value': np.max(metric_values),
                    'std_value': np.std(metric_values),
                    'run_count': len(metric_values),
                    'total_runs': len(data)
                })
        
        return pd.DataFrame(comparison_data)

def create_dataset_management_section(processor: BenchmarkProcessor):
    """Create the dataset management section"""
    st.header("📂 Dataset Management")
    
    # Display existing datasets
    if processor.datasets:
        st.subheader("Existing Datasets")
        
        for i, (label, dataset_info) in enumerate(processor.datasets.items()):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="dataset-card">
                    <h4>{label}</h4>
                    <p><strong>Runs:</strong> {dataset_info['summary']['total_runs']} | 
                    <strong>Models:</strong> {', '.join(dataset_info['summary']['models_list'])} |
                    <strong>Requests:</strong> {dataset_info['summary']['total_requests']:,}</p>
                    <small>Created: {dataset_info['created_at'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"Analyze", key=f"analyze_{i}"):
                    st.session_state.selected_dataset = label
                    st.session_state.analysis_mode = 'single'
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    processor.remove_dataset(label)
                    st.rerun()
    
    else:
        st.info("No datasets loaded yet. Add your first dataset below!")

def create_data_input_section(processor: BenchmarkProcessor):
    """Create the data input section"""
    st.header("📁 Add New Dataset")
    
    # Dataset label input
    dataset_label = st.text_input(
        "Dataset Label",
        placeholder="e.g., 'GPT-4 on Azure', 'Claude on AWS', 'Local LLaMA'",
        help="Give this dataset a meaningful name for comparison"
    )
    
    # Input method selection
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
        
        if uploaded_files and dataset_label:
            if st.button("Add Dataset", type="primary"):
                with st.spinner("Processing files..."):
                    data = processor.process_uploaded_files(uploaded_files, dataset_label)
                    if data:
                        success = processor.add_dataset(dataset_label, data)
                        if success:
                            st.success(f"Successfully added dataset '{dataset_label}' with {len(data)} runs!")
                            st.rerun()
                        else:
                            st.error("Failed to add dataset")
                    else:
                        st.error("No valid benchmark files found")
    
    else:
        directory_path = st.text_input(
            "Directory Path",
            placeholder="/path/to/benchmark/files",
            help="Enter the path to directory containing benchmark JSON files"
        )
        
        if directory_path and dataset_label:
            if st.button("Add Dataset", type="primary"):
                with st.spinner("Processing directory..."):
                    data = processor.process_directory(directory_path, dataset_label)
                    if data:
                        success = processor.add_dataset(dataset_label, data)
                        if success:
                            st.success(f"Successfully added dataset '{dataset_label}' with {len(data)} runs!")
                            st.rerun()
                        else:
                            st.error("Failed to add dataset")
                    else:
                        st.error("No benchmark files found in directory")

def create_metric_charts(processor: BenchmarkProcessor, dataset_label: str, metric: str):
    """Create charts for a specific metric from a single dataset"""
    data = processor.prepare_chart_data(dataset_label, metric)
    
    if data.empty:
        st.warning(f"No data available for {metric} in dataset '{dataset_label}'")
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
                title=f"Average {config['name']} by Run - {dataset_label}",
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
                title=f"Percentile Distribution - {config['name']} - {dataset_label}",
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
                title=f"Min/Max Range - {config['name']} - {dataset_label}",
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

def create_comparison_charts(processor: BenchmarkProcessor, dataset_labels: List[str], metric: str):
    """Create comparison charts for multiple datasets"""
    comparison_data = processor.prepare_comparison_data(dataset_labels, metric)
    
    if comparison_data.empty:
        st.warning(f"No comparison data available for {metric}")
        return
    
    config = processor.metrics_config[metric]
    
    # Create tabs for different comparison views
    tab1, tab2, tab3 = st.tabs(["📊 Average Comparison", "📈 Range Comparison", "📋 Summary Table"])
    
    with tab1:
        st.subheader(f"Average {config['name']} Comparison")
        
        fig = px.bar(
            comparison_data,
            x='dataset_label',
            y='avg_value',
            title=f"Average {config['name']} by Dataset",
            labels={
                'dataset_label': 'Dataset',
                'avg_value': f"Average {config['name']} ({config['unit']})"
            },
            color='dataset_label',
            color_discrete_sequence=processor.dataset_colors[:len(dataset_labels)]
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add percentage comparison
        if len(comparison_data) > 1:
            best_idx = comparison_data['avg_value'].idxmin()
            best_dataset = comparison_data.loc[best_idx, 'dataset_label']
            best_value = comparison_data.loc[best_idx, 'avg_value']
            
            st.markdown(f"""
            <div class="comparison-highlight">
                <strong>Best Performance:</strong> {best_dataset} with {best_value:.2f} {config['unit']}
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader(f"{config['name']} Range Comparison")
        
        fig = go.Figure()
        
        for i, row in comparison_data.iterrows():
            color = processor.dataset_colors[i % len(processor.dataset_colors)]
            
            # Add error bars showing min/max range
            fig.add_trace(go.Scatter(
                x=[row['dataset_label']],
                y=[row['avg_value']],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[row['max_value'] - row['avg_value']],
                    arrayminus=[row['avg_value'] - row['min_value']]
                ),
                mode='markers',
                marker=dict(size=10, color=color),
                name=row['dataset_label']
            ))
        
        fig.update_layout(
            title=f"{config['name']} Range Comparison (Min/Max)",
            xaxis_title="Dataset",
            yaxis_title=f"{config['name']} ({config['unit']})",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Detailed Comparison Table")
        
        # Format the comparison data for display
        display_data = comparison_data.copy()
        display_data['avg_value'] = display_data['avg_value'].round(2)
        display_data['min_value'] = display_data['min_value'].round(2)
        display_data['max_value'] = display_data['max_value'].round(2)
        display_data['std_value'] = display_data['std_value'].round(2)
        
        display_data = display_data.rename(columns={
            'dataset_label': 'Dataset',
            'avg_value': f'Average ({config["unit"]})',
            'min_value': f'Minimum ({config["unit"]})',
            'max_value': f'Maximum ({config["unit"]})',
            'std_value': f'Std Dev ({config["unit"]})',
            'run_count': 'Valid Runs',
            'total_runs': 'Total Runs'
        })
        
        st.dataframe(display_data, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = BenchmarkProcessor()
    
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = 'manage'
    
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    
    # Header
    st.title("📊 Advanced Benchmark Data Analyzer")
    st.markdown("Analyze and compare performance metrics across multiple datasets, models, and platforms")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🎯 Navigation")
        
        mode = st.radio(
            "Select Mode:",
            ["📂 Manage Datasets", "🔍 Single Dataset Analysis", "⚖️ Compare Datasets"],
            index=0 if st.session_state.analysis_mode == 'manage' else (1 if st.session_state.analysis_mode == 'single' else 2)
        )
        
        if mode == "📂 Manage Datasets":
            st.session_state.analysis_mode = 'manage'
        elif mode == "🔍 Single Dataset Analysis":
            st.session_state.analysis_mode = 'single'
        elif mode == "⚖️ Compare Datasets":
            st.session_state.analysis_mode = 'compare'
    
    # Main content based on selected mode
    if st.session_state.analysis_mode == 'manage':
        create_dataset_management_section(st.session_state.processor)
        st.divider()
        create_data_input_section(st.session_state.processor)
    
    elif st.session_state.analysis_mode == 'single':
        if not st.session_state.processor.datasets:
            st.warning("No datasets available. Please add datasets first.")
            st.session_state.analysis_mode = 'manage'
            st.rerun()
        
        # Dataset selection
        selected_dataset = st.selectbox(
            "Select dataset to analyze:",
            st.session_state.processor.get_dataset_labels(),
            index=0 if not st.session_state.selected_dataset else 
                  (st.session_state.processor.get_dataset_labels().index(st.session_state.selected_dataset) 
                   if st.session_state.selected_dataset in st.session_state.processor.get_dataset_labels() else 0)
        )
        
        if selected_dataset:
            st.session_state.selected_dataset = selected_dataset
            
            # Display dataset summary
            st.header(f"📈 Analysis: {selected_dataset}")
            summary = st.session_state.processor.get_dataset_summary(selected_dataset)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Runs", summary['total_runs'])
            with col2:
                st.metric("Unique Models", summary['unique_models'])
            with col3:
                st.metric("Total Requests", f"{summary['total_requests']:,}")
            with col4:
                st.metric("Avg Concurrency", f"{summary['avg_concurrency']:.1f}")
            
            # Metric selection and visualization
            st.subheader("📊 Metric Analysis")
            
            # Get available metrics for this dataset
            dataset_data = st.session_state.processor.get_dataset(selected_dataset)['data']
            available_metrics = [
                key for key in st.session_state.processor.metrics_config.keys()
                if any(key in run for run in dataset_data)
            ]
            
            selected_metric = st.selectbox(
                "Select metric to analyze:",
                available_metrics,
                format_func=lambda x: st.session_state.processor.metrics_config[x]['name']
            )
            
            if selected_metric:
                create_metric_charts(st.session_state.processor, selected_dataset, selected_metric)
    
    elif st.session_state.analysis_mode == 'compare':
        if len(st.session_state.processor.datasets) < 2:
            st.warning("You need at least 2 datasets to compare. Please add more datasets first.")
            st.session_state.analysis_mode = 'manage'
            st.rerun()
        
        st.header("⚖️ Dataset Comparison")
        
        # Dataset selection for comparison
        all_datasets = st.session_state.processor.get_dataset_labels()
        selected_datasets = st.multiselect(
            "Select datasets to compare:",
            all_datasets,
            default=all_datasets[:2] if len(all_datasets) >= 2 else all_datasets
        )
        
        if len(selected_datasets) >= 2:
            # Display comparison summary
            st.subheader("📋 Comparison Summary")
            
            comparison_cols = st.columns(len(selected_datasets))
            for i, dataset_label in enumerate(selected_datasets):
                with comparison_cols[i]:
                    summary = st.session_state.processor.get_dataset_summary(dataset_label)
                    st.markdown(f"""
                    <div class="dataset-card">
                        <h4>{dataset_label}</h4>
                        <p><strong>Runs:</strong> {summary['total_runs']}</p>
                        <p><strong>Models:</strong> {', '.join(summary['models_list'])}</p>
                        <p><strong>Requests:</strong> {summary['total_requests']:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Metric selection for comparison
            st.subheader("📊 Metric Comparison")
            
            # Get metrics available in all selected datasets
            common_metrics = None
            for dataset_label in selected_datasets:
                dataset_data = st.session_state.processor.get_dataset(dataset_label)['data']
                dataset_metrics = set(
                    key for key in st.session_state.processor.metrics_config.keys()
                    if any(key in run for run in dataset_data)
                )
                
                if common_metrics is None:
                    common_metrics = dataset_metrics
                else:
                    common_metrics = common_metrics.intersection(dataset_metrics)
            
            if common_metrics:
                selected_metric = st.selectbox(
                    "Select metric to compare:",
                    list(common_metrics),
                    format_func=lambda x: st.session_state.processor.metrics_config[x]['name']
                )
                
                if selected_metric:
                    create_comparison_charts(st.session_state.processor, selected_datasets, selected_metric)
            else:
                st.warning("No common metrics found across selected datasets.")
        
        else:
            st.info("Please select at least 2 datasets to compare.")
    
    # Footer with export options
    if st.session_state.processor.datasets:
        st.divider()
        st.header("💾 Data Export")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Download All Datasets"):
                all_data = {
                    label: {
                        'data': dataset_info['data'],
                        'summary': dataset_info['summary'],
                        'created_at': dataset_info['created_at'].isoformat()
                    }
                    for label, dataset_info in st.session_state.processor.datasets.items()
                }
                
                combined_json = json.dumps(all_data, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=combined_json,
                    file_name="all_benchmark_datasets.json",
                    mime="application/json"
                )
        
        with export_col2:
            if st.session_state.analysis_mode == 'compare' and 'selected_datasets' in locals() and len(selected_datasets) >= 2:
                if st.button("Download Comparison Report"):
                    # Create a comparison report
                    report_data = {
                        'comparison_summary': {
                            'datasets': selected_datasets,
                            'comparison_date': pd.Timestamp.now().isoformat(),
                            'metrics_analyzed': list(common_metrics) if 'common_metrics' in locals() else []
                        },
                        'dataset_summaries': {
                            label: st.session_state.processor.get_dataset_summary(label)
                            for label in selected_datasets
                        }
                    }
                    
                    # Add metric comparisons if a metric is selected
                    if 'selected_metric' in locals() and selected_metric:
                        comparison_data = st.session_state.processor.prepare_comparison_data(selected_datasets, selected_metric)
                        if not comparison_data.empty:
                            report_data['metric_comparison'] = {
                                'metric': selected_metric,
                                'metric_name': st.session_state.processor.metrics_config[selected_metric]['name'],
                                'unit': st.session_state.processor.metrics_config[selected_metric]['unit'],
                                'results': comparison_data.to_dict('records')
                            }
                    
                    report_json = json.dumps(report_data, indent=2, default=str)
                    st.download_button(
                        label="Download Report JSON",
                        data=report_json,
                        file_name="benchmark_comparison_report.json",
                        mime="application/json"
                    )

    # Help section
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        ### Getting Started
        1. **Add Datasets**: Upload benchmark files and give each set a meaningful label (e.g., "GPT-4 Azure", "Claude AWS")
        2. **Single Analysis**: Analyze individual datasets to understand performance patterns
        3. **Compare Datasets**: Select multiple datasets to compare average performance across platforms/models
        
        ### Dataset Labels
        Use descriptive labels that help you remember what each dataset represents:
        - Model names: "GPT-4", "Claude-3", "LLaMA-2"
        - Platforms: "Azure OpenAI", "AWS Bedrock", "Local Deployment"
        - Configurations: "High Concurrency", "Production Load", "Baseline Test"
        
        ### Comparison Features
        - **Average Comparison**: See which dataset performs best on average
        - **Range Comparison**: Understand variability with min/max ranges  
        - **Summary Tables**: Detailed statistics for deeper analysis
        
        ### Expected File Format
        Files should be named `*benchmark*_genai_perf.json` and contain:
        ```json
        {
          "time_to_first_token": {
            "unit": "ms",
            "avg": 11878.79,
            "p25": 10373.02,
            "p50": 12091.21,
            // ... other percentiles
          },
          "input_config": {
            "request_count": 50,
            "model": ["model_name"],
            "backend": "backend_name",
            "concurrency": 4
          }
        }
        ```
        """)

if __name__ == "__main__":
    main()