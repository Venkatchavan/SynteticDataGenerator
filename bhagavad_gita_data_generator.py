import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json
import base64
import os

# Set page configuration
st.set_page_config(
    page_title="Bhagavad Gita Data Generator",
    page_icon="🕉️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5722;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4CAF50;
    }
    .tooltip {
        color: #2196F3;
        font-size: 0.8rem;
        font-style: italic;
    }
    .important-note {
        background-color: #FFF9C4;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #E8F5E9;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .settings-section {
        background-color: #F3F4F6;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to save configuration for later use
def save_config(config_name, queries, sentiments, metric_ranges):
    """Save the current configuration for reuse"""
    configs_dir = "saved_configs"
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)
    
    config = {
        "name": config_name,
        "queries": queries,
        "sentiments": sentiments,
        "metric_ranges": metric_ranges
    }
    
    with open(f"{configs_dir}/{config_name}.json", "w") as f:
        json.dump(config, f)
    
    return True

# Function to load saved configuration
def load_config(config_name):
    """Load a previously saved configuration"""
    configs_dir = "saved_configs"
    try:
        with open(f"{configs_dir}/{config_name}.json", "r") as f:
            config = json.load(f)
        return config
    except:
        return None

def generate_synthetic_data(queries, sentiments, num_samples_per_query=10, date_range=30, metric_ranges=None):
    """
    Generate synthetic search data based on input queries and sentiments
    
    Parameters:
    - queries: List of search queries
    - sentiments: List of sentiment categories
    - num_samples_per_query: Number of data points per query
    - date_range: Number of days to span for timestamps
    - metric_ranges: Dictionary with min/max values for metrics
    """
    data = []
    
    # Use default metric ranges if none provided
    if metric_ranges is None:
        metric_ranges = {
            "click_through_rate": {"min": 0.01, "max": 0.2},
            "bounce_rate": {"min": 0.2, "max": 0.9},
            "avg_session_duration": {"min": 10, "max": 300}
        }
    
    # Generate timestamps within the specified date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=date_range)
    
    for query in queries:
        for _ in range(num_samples_per_query):
            # Random timestamp within the date range
            random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
            timestamp = start_date + timedelta(seconds=random_seconds)
            
            # Randomly select a sentiment for each data point
            sentiment = random.choice(sentiments)
            
            # Generate random metrics based on provided ranges
            click_through_rate = round(random.uniform(
                metric_ranges["click_through_rate"]["min"], 
                metric_ranges["click_through_rate"]["max"]
            ), 3)
            
            bounce_rate = round(random.uniform(
                metric_ranges["bounce_rate"]["min"], 
                metric_ranges["bounce_rate"]["max"]
            ), 3)
            
            avg_session_duration = round(random.uniform(
                metric_ranges["avg_session_duration"]["min"], 
                metric_ranges["avg_session_duration"]["max"]
            ), 1)
            
            data.append({
                'query': query,
                'timestamp': timestamp,
                'sentiment': sentiment,
                'click_through_rate': click_through_rate,
                'bounce_rate': bounce_rate,
                'avg_session_duration': avg_session_duration
            })
    
    return pd.DataFrame(data)

def generate_excel_download_link(df):
    """Generate a link to download the dataframe as an Excel file"""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Data')
    writer.close()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="bhagavad_gita_search_data.xlsx">Download Excel File</a>'

def main():
    # Main header with emoji
    st.markdown('<h1 class="main-header">🕉️ Bhagavad Gita Search Data Generator</h1>', unsafe_allow_html=True)
    
    # App description
    with st.expander("ℹ️ About this app", expanded=True):
        st.markdown("""
        This application generates synthetic search data related to the Bhagavad Gita. 
        You can customize the search queries, sentiment categories, and the number of data points to generate.
        
        <div class="important-note">
        <strong>How to use:</strong>
        <ol>
            <li>Enter or select your desired search queries</li>
            <li>Choose sentiment categories</li>
            <li>Set the number of samples per query</li>
            <li>Customize metric ranges (optional)</li>
            <li>Click "Generate Synthetic Data"</li>
            <li>View and download the generated data</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Default queries related to Bhagavad Gita
    default_queries = [
        "krishna teachings", 
        "arjuna dilemma", 
        "dharma meaning", 
        "karma yoga", 
        "bhagavad gita chapter 2", 
        "yoga of knowledge",
        "detachment bhagavad gita",
        "gita on duty",
        "krishna's divine form",
        "path of devotion"
    ]
    
    # Default sentiments
    default_sentiments = ["positive", "negative", "neutral", "curious", "inspired"]
    
    # Default metric ranges
    default_metric_ranges = {
        "click_through_rate": {"min": 0.01, "max": 0.2},
        "bounce_rate": {"min": 0.2, "max": 0.9},
        "avg_session_duration": {"min": 10, "max": 300}
    }
    
    # Load saved configurations if available
    saved_configs_dir = "saved_configs"
    saved_configs = []
    if os.path.exists(saved_configs_dir):
        saved_configs = [f.replace(".json", "") for f in os.listdir(saved_configs_dir) if f.endswith(".json")]
    
    # Option to load saved configuration
    if saved_configs:
        with st.expander("💾 Load Saved Configuration", expanded=False):
            selected_config = st.selectbox("Select a saved configuration", [""] + saved_configs)
            if selected_config:
                config = load_config(selected_config)
                if config:
                    st.success(f"✅ Configuration '{selected_config}' loaded successfully!")
                    default_queries = config["queries"]
                    default_sentiments = config["sentiments"]
                    default_metric_ranges = config["metric_ranges"]
    
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="sub-header">📝 Search Queries</h2>', unsafe_allow_html=True)
        
        # Help text with tooltip
        st.markdown('<p class="tooltip">Enter one query per line. These represent what users might search for.</p>', 
                   unsafe_allow_html=True)
        
        # Custom queries input
        user_queries = st.text_area(
            "Enter search queries", 
            "\n".join(default_queries), 
            height=150
        )
        queries = [q.strip() for q in user_queries.split("\n") if q.strip()]
        
        # Display queries as a dropdown for selection
        selected_queries = st.multiselect(
            "Select queries to include", 
            queries,
            default=queries
        )
        
        # Number of samples per query
        samples_per_query = st.slider(
            "Number of samples per query", 
            min_value=1, 
            max_value=50, 
            value=10,
            help="More samples will generate more data points for each query"
        )
    
    with col2:
        st.markdown('<h2 class="sub-header">😊 Sentiment Categories</h2>', unsafe_allow_html=True)
        
        # Help text with tooltip
        st.markdown('<p class="tooltip">Enter one sentiment per line. These represent different emotional responses to search results.</p>', 
                   unsafe_allow_html=True)
        
        # Custom sentiments input
        user_sentiments = st.text_area(
            "Enter sentiment categories", 
            "\n".join(default_sentiments), 
            height=150
        )
        sentiments = [s.strip() for s in user_sentiments.split("\n") if s.strip()]
        
        # Display sentiments as a dropdown for selection
        selected_sentiments = st.multiselect(
            "Select sentiments to include", 
            sentiments,
            default=sentiments
        )
        
        # Date range option
        st.markdown("### ⏱️ Time Range")
        days_range = st.slider(
            "Days of data to generate", 
            min_value=1, 
            max_value=90, 
            value=30,
            help="The generated data will span this many days into the past"
        )
    
    # Advanced metrics customization
    with st.expander("⚙️ Advanced Metrics Settings", expanded=False):
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### Customize Metric Ranges")
        st.markdown("Adjust the minimum and maximum values for each metric to simulate different scenarios.")
        
        # Customizing click-through rate
        ctr_col1, ctr_col2 = st.columns(2)
        with ctr_col1:
            ctr_min = st.number_input(
                "Click-Through Rate (Min)", 
                min_value=0.001, 
                max_value=0.5, 
                value=default_metric_ranges["click_through_rate"]["min"],
                format="%.3f",
                step=0.001
            )
        with ctr_col2:
            ctr_max = st.number_input(
                "Click-Through Rate (Max)", 
                min_value=0.001, 
                max_value=1.0, 
                value=default_metric_ranges["click_through_rate"]["max"],
                format="%.3f",
                step=0.001
            )
        
        # Customizing bounce rate
        br_col1, br_col2 = st.columns(2)
        with br_col1:
            br_min = st.number_input(
                "Bounce Rate (Min)", 
                min_value=0.0, 
                max_value=1.0, 
                value=default_metric_ranges["bounce_rate"]["min"],
                format="%.2f",
                step=0.01
            )
        with br_col2:
            br_max = st.number_input(
                "Bounce Rate (Max)", 
                min_value=0.0, 
                max_value=1.0, 
                value=default_metric_ranges["bounce_rate"]["max"],
                format="%.2f",
                step=0.01
            )
        
        # Customizing session duration
        sd_col1, sd_col2 = st.columns(2)
        with sd_col1:
            sd_min = st.number_input(
                "Session Duration (Min)", 
                min_value=1, 
                max_value=1000, 
                value=int(default_metric_ranges["avg_session_duration"]["min"]),
                step=1
            )
        with sd_col2:
            sd_max = st.number_input(
                "Session Duration (Max)", 
                min_value=1, 
                max_value=3600, 
                value=int(default_metric_ranges["avg_session_duration"]["max"]),
                step=10
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Option to save current configuration
    with st.expander("💾 Save Current Configuration", expanded=False):
        config_name = st.text_input("Configuration Name", "my_bhagavad_gita_config")
        save_config_button = st.button("Save Configuration")
        
        if save_config_button:
            # Create metric ranges dictionary
            metric_ranges = {
                "click_through_rate": {"min": ctr_min, "max": ctr_max},
                "bounce_rate": {"min": br_min, "max": br_max},
                "avg_session_duration": {"min": sd_min, "max": sd_max}
            }
            
            # Save configuration
            if save_config(config_name, selected_queries, selected_sentiments, metric_ranges):
                st.success(f"✅ Configuration saved as '{config_name}'")
    
    # Generate data section (centered)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        generate_button = st.button("🚀 Generate Synthetic Data", use_container_width=True)
    
    # Preview feature
    preview_container = st.empty()
    
    # Generate data on button click
    if generate_button:
        if not selected_queries:
            st.error("⚠️ Please select at least one query.")
        elif not selected_sentiments:
            st.error("⚠️ Please select at least one sentiment category.")
        else:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create metric ranges dictionary
            metric_ranges = {
                "click_through_rate": {"min": ctr_min, "max": ctr_max},
                "bounce_rate": {"min": br_min, "max": br_max},
                "avg_session_duration": {"min": sd_min, "max": sd_max}
            }
            
            # Simulate processing time for large datasets
            total_items = len(selected_queries) * samples_per_query
            
            # Generate a small preview first
            status_text.text("Generating preview...")
            preview_df = generate_synthetic_data(
                selected_queries[:2] if len(selected_queries) > 1 else selected_queries, 
                selected_sentiments, 
                min(3, samples_per_query),
                days_range,
                metric_ranges
            )
            
            # Show preview
            with preview_container.container():
                st.subheader("🔍 Data Preview")
                st.dataframe(preview_df.head())
                
                # Ask user if they want to proceed
                st.info(f"Preview shown above. Proceeding will generate {total_items} records.")
                
            # Generate the full dataset
            status_text.text("Generating full dataset...")
            
            # Simulate progress
            for i in range(100):
                time.sleep(0.01)  # Simulate work being done
                progress_bar.progress(i + 1)
            
            # Generate the actual data
            df = generate_synthetic_data(
                selected_queries, 
                selected_sentiments, 
                samples_per_query,
                days_range,
                metric_ranges
            )
            
            status_text.text(f"✅ Generated {len(df)} records successfully!")
            
            # Convert timestamp to datetime object for proper plotting
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create tabs for data display and analytics
            data_tab, analytics_tab, report_tab = st.tabs(["Generated Data", "Interactive Analytics", "Data Report"])
            
            with data_tab:
                # Main container for data
                st.subheader(f"📊 Generated Data ({len(df)} entries)")
                st.dataframe(df, use_container_width=True)
                
                # Download options in columns
                st.markdown("### 📥 Download Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="bhagavad_gita_search_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Add JSON export option
                    json_data = df.to_json(orient="records")
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json_data,
                        file_name="bhagavad_gita_search_data.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    # Add Excel export option
                    st.markdown(generate_excel_download_link(df), unsafe_allow_html=True)
            
            with analytics_tab:
                # Display interactive analytics with Plotly
                st.subheader("📈 Interactive Data Analytics")
                
                # Query selector for filtering
                selected_analysis_query = st.selectbox(
                    "Select a query to analyze",
                    ["All Queries"] + selected_queries
                )
                
                # Filter data based on selection
                if selected_analysis_query != "All Queries":
                    filtered_df = df[df['query'] == selected_analysis_query]
                else:
                    filtered_df = df
                
                # Create two columns for top visualizations
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Sentiment distribution pie chart
                    st.markdown("### 😊 Sentiment Distribution")
                    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    
                    fig = px.pie(
                        sentiment_counts, 
                        values='Count', 
                        names='Sentiment',
                        title=f"Sentiment Distribution for {selected_analysis_query}",
                        color_discrete_sequence=px.colors.sequential.Viridis
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                    
                with viz_col2:
                    # Metrics comparison by sentiment
                    st.markdown("### 📊 Metrics by Sentiment")
                    
                    # Group by sentiment and calculate averages
                    metrics_by_sentiment = filtered_df.groupby('sentiment')[
                        ['click_through_rate', 'bounce_rate', 'avg_session_duration']
                    ].mean().reset_index()
                    
                    # Choose which metric to display
                    selected_metric = st.selectbox(
                        "Select metric to display",
                        ["click_through_rate", "bounce_rate", "avg_session_duration"]
                    )
                    
                    # Create bar chart
                    fig = px.bar(
                        metrics_by_sentiment,
                        x='sentiment',
                        y=selected_metric,
                        title=f"Average {selected_metric.replace('_', ' ').title()} by Sentiment",
                        color='sentiment',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Time series analysis
                st.markdown("### ⏱️ Time Series Analysis")
                
                # Group by date for time series
                filtered_df['date'] = filtered_df['timestamp'].dt.date
                time_series_data = filtered_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
                
                # Line chart for trends over time
                fig = px.line(
                    time_series_data,
                    x='date',
                    y='count',
                    color='sentiment',
                    title="Query Sentiment Trends Over Time",
                    labels={'count': 'Number of Queries', 'date': 'Date'}
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Query Count")
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap of query patterns
                st.markdown("### 🗓️ Query Pattern Heatmap")

                # Add hour of day information
                filtered_df['hour'] = filtered_df['timestamp'].dt.hour
                filtered_df['day_of_week'] = filtered_df['timestamp'].dt.day_name()

                # Prepare data for heatmap
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                hour_counts = filtered_df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')

                # Create pivot table for heatmap
                heatmap_data = pd.pivot_table(
                    hour_counts, 
                    values='count', 
                    index='day_of_week',
                    columns='hour',
                    fill_value=0
                )

                # Reorder days
                heatmap_data = heatmap_data.reindex(day_order)

                # Make sure all hours from 0-23 are represented in the columns
                for hour in range(24):
                    if hour not in heatmap_data.columns:
                        heatmap_data[hour] = 0
                heatmap_data = heatmap_data.sort_index(axis=1)

                # Create heatmap
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Query Count"),
                    x=heatmap_data.columns.tolist(),  # Use actual columns instead of range(24)
                    y=day_order,
                    color_continuous_scale="Viridis",
                    title="Query Activity Heatmap (Day of Week vs. Hour)"
                )
                fig.update_layout(
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation analysis
                st.markdown("### 🔄 Correlation Between Metrics")
                
                # Correlation matrix
                corr_matrix = filtered_df[['click_through_rate', 'bounce_rate', 'avg_session_duration']].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Correlation Matrix of Metrics",
                    labels=dict(color="Correlation")
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with report_tab:
                # Generate a comprehensive report
                st.subheader("📋 Data Summary Report")
                
                # Overall statistics
                st.markdown("### 📊 Overall Statistics")
                
                # Create a high-level summary
                total_queries = len(df)
                unique_queries = df['query'].nunique()
                date_range = (df['timestamp'].min().date(), df['timestamp'].max().date())
                
                # Display metrics in cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Data Points", total_queries)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Unique Queries", unique_queries)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Date Range", f"{date_range[0]} to {date_range[1]}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Summary by query
                st.markdown("### 🔍 Query Summary")
                
                # Summary table by query
                query_summary = df.groupby('query').agg({
                    'timestamp': 'count',
                    'click_through_rate': 'mean',
                    'bounce_rate': 'mean',
                    'avg_session_duration': 'mean'
                }).reset_index()
                
                query_summary.columns = ['Query', 'Count', 'Avg CTR', 'Avg Bounce Rate', 'Avg Session Duration (s)']
                query_summary = query_summary.sort_values('Count', ascending=False)
                
                # Format numbers for better readability
                query_summary['Avg CTR'] = query_summary['Avg CTR'].map(lambda x: f"{x:.3f}")
                query_summary['Avg Bounce Rate'] = query_summary['Avg Bounce Rate'].map(lambda x: f"{x:.3f}")
                query_summary['Avg Session Duration (s)'] = query_summary['Avg Session Duration (s)'].map(lambda x: f"{x:.1f}")
                
                st.dataframe(query_summary, use_container_width=True)
                
                # Summary by sentiment
                st.markdown("### 😊 Sentiment Performance")
                
                # Summary table by sentiment
                sentiment_summary = df.groupby('sentiment').agg({
                    'timestamp': 'count',
                    'click_through_rate': 'mean',
                    'bounce_rate': 'mean',
                    'avg_session_duration': 'mean'
                }).reset_index()
                
                sentiment_summary.columns = ['Sentiment', 'Count', 'Avg CTR', 'Avg Bounce Rate', 'Avg Session Duration (s)']
                
                # Format numbers for better readability
                sentiment_summary['Avg CTR'] = sentiment_summary['Avg CTR'].map(lambda x: f"{x:.3f}")
                sentiment_summary['Avg Bounce Rate'] = sentiment_summary['Avg Bounce Rate'].map(lambda x: f"{x:.3f}")
                sentiment_summary['Avg Session Duration (s)'] = sentiment_summary['Avg Session Duration (s)'].map(lambda x: f"{x:.1f}")
                
                st.dataframe(sentiment_summary, use_container_width=True)
                
                # Key insights
                st.markdown("### 💡 Key Insights")
                
                # Best and worst performing queries by CTR
                best_query = query_summary.iloc[query_summary['Avg CTR'].astype(float).argmax()]['Query']
                worst_query = query_summary.iloc[query_summary['Avg CTR'].astype(float).argmin()]['Query']
                
                # Best performing sentiment
                best_sentiment = sentiment_summary.iloc[sentiment_summary['Avg CTR'].astype(float).argmax()]['Sentiment']
                
                # Generate insights
                insights = [
                    f"- The best performing query is '**{best_query}**' with the highest average CTR.",
                    f"- The query with the lowest performance is '**{worst_query}**'.",
                    f"- Searches with '**{best_sentiment}**' sentiment tend to have the highest engagement.",
                    f"- The dataset contains {total_queries} data points across {unique_queries} different queries.",
                    f"- The data spans from {date_range[0]} to {date_range[1]}."
                ]
                
                for insight in insights:
                    st.markdown(insight)
                
                # Download report button
                st.markdown("### 📥 Download Report")
                
                # Generate a detailed report in HTML format
                report_html = f"""
                <html>
                <head>
                    <title>Bhagavad Gita Search Data Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 20px; }}
                        h1 {{ color: #FF5722; }}
                        h2 {{ color: #4CAF50; margin-top: 30px; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .insight {{ background-color: #fff9c4; padding: 10px; border-radius: 5px; }}
                    </style>
                </head>
                <body>
                    <h1>Bhagavad Gita Search Data Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Overview</h2>
                    <p>This report provides an analysis of synthetic search data related to the Bhagavad Gita.</p>
                    <ul>
                        <li>Total data points: {total_queries}</li>
                        <li>Unique queries: {unique_queries}</li>
                        <li>Date range: {date_range[0]} to {date_range[1]}</li>
                    </ul>
                    
                    <h2>Query Summary</h2>
                    {query_summary.to_html()}
                    
                    <h2>Sentiment Performance</h2>
                    {sentiment_summary.to_html()}
                    
                    <h2>Key Insights</h2>
                    <div class="insight">
                        <ul>
                            {"".join(f"<li>{insight[2:]}</li>" for insight in insights)}
                        </ul>
                    </div>
                </body>
                </html>
                """
                
                # Convert HTML to downloadable format
                report_b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{report_b64}" download="bhagavad_gita_search_report.html">📥 Download HTML Report</a>'
                st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()