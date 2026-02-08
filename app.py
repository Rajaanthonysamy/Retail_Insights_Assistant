"""
Streamlit UI for Retail Insights Assistant
Multi-Agent GenAI System for Sales Analytics
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from orchestrator import RetailInsightsOrchestrator
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Retail Insights Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: transparent;
        padding: 1rem 0;
        border-left: 3px solid #1f77b4;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .agent-status {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }

    /* Right sidebar panel styling */
    .right-panel {
        background-color: #f0f2f6;
        padding: 1rem;
        border-left: 1px solid #e0e0e0;
        border-radius: 8px;
        min-height: 400px;
        margin-top: -2rem;
    }

</style>
""", unsafe_allow_html=True)


def get_api_key():
    """Get API key from session state only (user must enter it via UI)"""
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    return None


@st.cache_resource
def initialize_orchestrator(_api_key):
    """Initialize the multi-agent orchestrator (cached per API key)"""
    model_name = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
    data_path = os.getenv("DATA_PATH", "Sales Dataset/")

    return RetailInsightsOrchestrator(
        api_key=_api_key,
        model_name=model_name,
        data_path=data_path
    )


def render_agent_status_native(container, current_agent=0, error_agent=None):
    """Render agent status using native Streamlit components"""
    agents = [
        ("🔍 Query Resolution", "Analyzes user query & generates SQL"),
        ("📊 Data Extraction", "Executes SQL & retrieves data"),
        ("✅ Validation", "Validates data quality"),
        ("💬 Response Generation", "Generates insights")
    ]

    with container.container():
        # Container with border and styling
        st.markdown("<div style='border: 2px solid #667eea; border-radius: 10px; padding: 15px; background: #f8f9fa;'>", unsafe_allow_html=True)

        # Show idle state or processing state
        if current_agent == 0:
            # Idle/Ready state
            st.markdown("""
            <div style='text-align: center; padding: 30px 10px; color: #666;'>
                <div style='font-size: 48px; margin-bottom: 10px;'>💤</div>
                <div style='font-size: 16px; font-weight: bold; color: #667eea;'>Ready</div>
                <div style='font-size: 12px; margin-top: 5px;'>Waiting for your query...</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Processing state - show all agents
            for i, (name, description) in enumerate(agents, 1):
                # Determine status
                if error_agent and i == error_agent:
                    icon = '🔴'
                    status_text = 'ERROR'
                    status_color = '#dc3545'
                    bg_color = '#ffe6e6'
                elif i < current_agent:
                    icon = '✅'
                    status_text = 'DONE'
                    status_color = '#28a745'
                    bg_color = '#e6f7e6'
                elif i == current_agent:
                    icon = '🟢'
                    status_text = 'RUNNING'
                    status_color = '#ffc107'
                    bg_color = '#fff9e6'
                else:
                    icon = '⚪'
                    status_text = 'WAITING'
                    status_color = '#6c757d'
                    bg_color = '#f8f9fa'

                # Agent card - MUST be inside the loop
                st.markdown(f"""
                <div style='background: {bg_color}; padding: 10px; margin: 6px 0;
                            border-left: 4px solid {status_color}; border-radius: 5px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='font-size: 18px;'>{icon}</span>
                            <strong style='margin-left: 5px; font-size: 13px;'>Agent {i}</strong>
                        </div>
                        <span style='background: {status_color}; color: white; padding: 2px 8px;
                                     border-radius: 8px; font-size: 10px; font-weight: bold;'>
                            {status_text}
                        </span>
                    </div>
                    <div style='margin-top: 4px; padding-left: 25px;'>
                        <div style='font-weight: 600; font-size: 12px; color: #333;'>{name}</div>
                        <div style='font-size: 10px; color: #666;'>{description}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def display_agent_workflow(metadata: dict):
    """Display agent workflow status"""
    st.markdown("### 🤖 Multi-Agent Workflow")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        query_status = metadata.get('query_resolution', {})
        status_icon = "✅" if query_status.get('success') else "❌"
        st.markdown(f"""
        **Query Resolution Agent**
        {status_icon} {'Completed' if query_status.get('success') else 'Failed'}
        """)

    with col2:
        extraction_status = metadata.get('data_extraction', {})
        status_icon = "✅" if extraction_status.get('success') else "❌"
        records = extraction_status.get('records_extracted', 0)
        st.markdown(f"""
        **Data Extraction Agent**
        {status_icon} {'Completed' if extraction_status.get('success') else 'Failed'}
        Records: {records}
        """)

    with col3:
        validation_status = metadata.get('validation', {})
        status_icon = "✅" if validation_status.get('success') else "❌"
        confidence = validation_status.get('confidence', 0)
        st.markdown(f"""
        **Validation Agent**
        {status_icon} {'Completed' if validation_status.get('success') else 'Failed'}
        Confidence: {confidence:.0%}
        """)

    with col4:
        response_status = metadata.get('response_generation', {})
        status_icon = "✅" if response_status.get('success') else "❌"
        st.markdown(f"""
        **Response Generation Agent**
        {status_icon} {'Completed' if response_status.get('success') else 'Failed'}
        """)


def display_summary_visualizations(data: dict):
    """Display visualizations for summary data"""
    if not data or data.get('data_type') != 'summary':
        return

    # Top Categories Chart
    if 'top_categories' in data and data['top_categories']:
        st.markdown("### 📈 Top Performing Categories")
        df_categories = pd.DataFrame(data['top_categories'])

        if not df_categories.empty and 'Category' in df_categories.columns:
            fig = px.bar(
                df_categories.head(10),
                x='Category',
                y='total_revenue',
                title='Top 10 Categories by Revenue',
                labels={'total_revenue': 'Total Revenue (INR)', 'Category': 'Category'},
                color='total_revenue',
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Regional Performance Chart
    if 'regional_performance' in data and data['regional_performance']:
        st.markdown("### 🗺️ Regional Performance")
        df_regions = pd.DataFrame(data['regional_performance'])

        if not df_regions.empty and 'state' in df_regions.columns:
            fig = px.bar(
                df_regions.head(10),
                x='state',
                y='total_revenue',
                title='Top 10 States by Revenue',
                labels={'total_revenue': 'Total Revenue (INR)', 'state': 'State'},
                color='order_count',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">📊 Retail Insights Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Agent GenAI System for Sales Analytics</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # API Key Input
        st.markdown("### 🔑 OpenAI API Key")
        api_key = get_api_key()

        if api_key:
            # Key already set - show masked version
            masked = api_key[:4] + "..." + api_key[-4:]
            st.success(f"API Key active: `{masked}`")
            if st.button("🔄 Change API Key", key="change_key"):
                st.session_state.openai_api_key = ""
                initialize_orchestrator.clear()
                st.rerun()
        else:
            # No key yet - user must enter it
            key_input = st.text_input(
                "Enter your OpenAI API Key:",
                type="password",
                placeholder="sk-...",
                key="api_key_input"
            )
            if st.button("✅ Set API Key", type="primary", key="set_key"):
                if key_input and len(key_input) > 10:
                    st.session_state.openai_api_key = key_input
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")
            st.info("Key is stored in memory only for this session.")

        st.markdown("---")

        mode = st.radio(
            "Select Mode:",
            ["💬 Conversational Q&A", "📋 Generate Summary"],
            help="Choose between asking specific questions or generating a comprehensive summary"
        )

        st.markdown("---")
        # st.markdown("### 📖 About")
        # st.markdown("""
        # This is a **multi-agent GenAI system** that:
        # - 🤖 Uses 4 specialized agents
        # - 📊 Analyzes sales data with DuckDB
        # - 🧠 Powered by LangGraph & OpenAI
        # - ⚡ Scales to 100GB+ datasets
        # """)

        # st.markdown("---")
        # st.markdown("### 🤖 Agent Pipeline")

        # with st.expander("🔍 **Agent 1: Query Resolution**", expanded=False):
        #     st.markdown("""
        #     **Purpose:** Converts natural language to SQL

        #     **Tasks:**
        #     - Analyzes user intent
        #     - Maps to database schema
        #     - Generates optimized SQL queries
        #     - Handles date parsing & filters
        #     """)

        # with st.expander("📊 **Agent 2: Data Extraction**", expanded=False):
        #     st.markdown("""
        #     **Purpose:** Executes queries & retrieves data

        #     **Tasks:**
        #     - Runs SQL on DuckDB
        #     - Handles query errors
        #     - Provides fallback data
        #     - Adds dataset metadata
        #     """)

        # with st.expander("✅ **Agent 3: Validation**", expanded=False):
        #     st.markdown("""
        #     **Purpose:** Validates results quality

        #     **Tasks:**
        #     - Checks data completeness
        #     - Validates against query intent
        #     - Assigns confidence scores
        #     - Identifies data limitations
        #     """)

        # with st.expander("💬 **Agent 4: Response Generation**", expanded=False):
        #     st.markdown("""
        #     **Purpose:** Creates insights & recommendations

        #     **Tasks:**
        #     - Generates business insights
        #     - Explains data limitations
        #     - Provides actionable advice
        #     - Formats professional responses
        #     """)

        # st.markdown("---")
        st.markdown("### 🔍 Example Questions")
        st.markdown("""
        - Which category has the highest sales?
        - What is the total revenue by state?
        - Show me top performing products
        - Which region saw the most orders?
        - What are the sales trends?
        """)

    # Initialize orchestrator (only when API key is available)
    api_key = get_api_key()
    if not api_key:
        st.warning("🔑 Please enter your OpenAI API Key in the sidebar to get started.")
        st.stop()

    try:
        orchestrator = initialize_orchestrator(api_key)
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        st.stop()

    # Main content
    if mode == "💬 Conversational Q&A":
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'agent_status' not in st.session_state:
            st.session_state.agent_status = 0  # 0 = ready, 1-4 = processing

        # Two-column layout: Main Chat (left) | Agent Sidebar (right)
        col_main, col_sidebar = st.columns([7, 3])

        # ==========================================
        # RIGHT SIDEBAR - Agent Pipeline
        # ==========================================
        with col_sidebar:
            # st.markdown('<div class="right-panel">', unsafe_allow_html=True)

            st.markdown("## 🔄 Agent Pipeline")
            st.markdown("---")

            # Live Status Section
            st.markdown("### 📡 Live Status")
            status_placeholder = st.empty()
            render_agent_status_native(status_placeholder, st.session_state.agent_status)

            st.markdown("---")

            # Agent Details (collapsible - same as left sidebar)
            st.markdown("### 🤖 Agent Details")

            with st.expander("🔍 **Agent 1: Query Resolution**", expanded=False):
                st.markdown("""
                **Purpose:** Converts natural language to SQL

                **Tasks:**
                - Analyzes user intent
                - Maps to database schema
                - Generates optimized SQL
                - Handles date parsing & filters
                """)

            with st.expander("📊 **Agent 2: Data Extraction**", expanded=False):
                st.markdown("""
                **Purpose:** Executes queries & retrieves data

                **Tasks:**
                - Runs SQL on DuckDB
                - Handles query errors
                - Provides fallback data
                - Adds dataset metadata
                """)

            with st.expander("✅ **Agent 3: Validation**", expanded=False):
                st.markdown("""
                **Purpose:** Validates results quality

                **Tasks:**
                - Checks data completeness
                - Validates against query intent
                - Assigns confidence scores
                - Identifies data limitations
                """)

            with st.expander("💬 **Agent 4: Response Generation**", expanded=False):
                st.markdown("""
                **Purpose:** Creates insights & recommendations

                **Tasks:**
                - Generates business insights
                - Explains data limitations
                - Provides actionable advice
                - Formats professional responses
                """)

            st.markdown('</div>', unsafe_allow_html=True)

        # ==========================================
        # MAIN CONTENT - Chat Interface (left side)
        # ==========================================
        with col_main:
            st.markdown("## 💬 Chat with Your Sales Data")

            # Display chat history (chronological order - oldest first)
            if st.session_state.chat_history:
                for chat in st.session_state.chat_history:
                    # User message
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-left: 20%;">
                        <strong>🙋 You:</strong><br>
                        {chat['query']}
                        <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">{chat['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # AI response
                    st.markdown(f"""
                    <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-right: 20%;">
                        <strong>🤖 Assistant:</strong><br>
                        {chat['result']['response']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Expandable details
                    with st.expander("🔍 View Details"):
                        # Show SQL query if available
                        if chat['result'].get('sql_query'):
                            st.markdown("### 🔍 Generated SQL Query")
                            st.code(chat['result']['sql_query'], language='sql')
                            st.markdown("---")

                        display_agent_workflow(chat['result'].get('metadata', {}))

                        # Show extracted data if available
                        if chat['result'].get('data'):
                            st.markdown("### 📊 Extracted Data")
                            data = chat['result']['data']

                            if data.get('data_type') == 'query_result' and data.get('query_result'):
                                df = pd.DataFrame(data['query_result'])

                                # Add tabs for Table and Chart view
                                tab1, tab2 = st.tabs(["📋 Table View", "📈 Chart View"])

                                with tab1:
                                    st.dataframe(df, use_container_width=True)

                                with tab2:
                                    # Dynamic chart selection based on data characteristics
                                    try:
                                        # Detect column types
                                        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns.tolist()
                                        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                                        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower() or 'year' in col.lower()]

                                        num_rows = len(df)

                                        # Decision tree for chart type selection
                                        chart_created = False

                                        # 1. TIME SERIES: Line chart for temporal data
                                        if date_cols and numeric_cols:
                                            date_col = date_cols[0]
                                            num_col = numeric_cols[0]

                                            fig = px.line(
                                                df,
                                                x=date_col,
                                                y=num_col,
                                                title=f"{num_col} Over Time",
                                                labels={date_col: date_col, num_col: num_col},
                                                markers=True
                                            )
                                            fig.update_layout(hovermode='x unified')
                                            st.plotly_chart(fig, use_container_width=True)
                                            chart_created = True

                                        # 2. PROPORTIONS: Pie chart for 2-8 categories
                                        elif len(categorical_cols) > 0 and len(numeric_cols) > 0 and 2 <= num_rows <= 8:
                                            cat_col = categorical_cols[0]
                                            num_col = numeric_cols[0]

                                            fig = px.pie(
                                                df,
                                                names=cat_col,
                                                values=num_col,
                                                title=f"{num_col} Distribution by {cat_col}",
                                                hole=0.3  # Donut chart
                                            )
                                            fig.update_traces(textposition='inside', textinfo='percent+label')
                                            st.plotly_chart(fig, use_container_width=True)
                                            chart_created = True

                                        # 3. RANKINGS: Horizontal bar chart for many categories (better readability)
                                        elif len(categorical_cols) > 0 and len(numeric_cols) > 0 and num_rows > 8:
                                            cat_col = categorical_cols[0]
                                            num_col = numeric_cols[0]

                                            # Sort by numeric value for ranking
                                            df_sorted = df.nlargest(15, num_col)

                                            fig = px.bar(
                                                df_sorted,
                                                y=cat_col,
                                                x=num_col,
                                                orientation='h',
                                                title=f"Top 15: {num_col} by {cat_col}",
                                                labels={cat_col: cat_col, num_col: num_col},
                                                color=num_col,
                                                color_continuous_scale='blues'
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                            chart_created = True

                                        # 4. COMPARISON: Vertical bar chart for fewer categories
                                        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
                                            cat_col = categorical_cols[0]
                                            num_col = numeric_cols[0]

                                            fig = px.bar(
                                                df.head(10),
                                                x=cat_col,
                                                y=num_col,
                                                title=f"{num_col} by {cat_col}",
                                                labels={cat_col: cat_col, num_col: num_col},
                                                color=num_col,
                                                color_continuous_scale='viridis'
                                            )
                                            fig.update_layout(xaxis_tickangle=-45)
                                            st.plotly_chart(fig, use_container_width=True)
                                            chart_created = True

                                        # 5. MULTI-METRIC: Grouped bar chart for multiple metrics
                                        elif len(numeric_cols) >= 2:
                                            first_col = df.columns[0]

                                            fig = px.bar(
                                                df.head(10),
                                                x=first_col,
                                                y=numeric_cols[:3],  # Max 3 metrics
                                                title="Multi-Metric Comparison",
                                                barmode='group',
                                                labels={first_col: first_col}
                                            )
                                            fig.update_layout(xaxis_tickangle=-45)
                                            st.plotly_chart(fig, use_container_width=True)
                                            chart_created = True

                                        # 6. CORRELATION: Scatter plot for 2 numeric columns with categorical
                                        elif len(numeric_cols) == 2 and len(categorical_cols) > 0:
                                            cat_col = categorical_cols[0]

                                            fig = px.scatter(
                                                df.head(50),
                                                x=numeric_cols[0],
                                                y=numeric_cols[1],
                                                color=cat_col,
                                                size=numeric_cols[0],
                                                title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                                                labels={numeric_cols[0]: numeric_cols[0], numeric_cols[1]: numeric_cols[1]}
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                            chart_created = True

                                        # 7. SINGLE VALUE: Display as metric
                                        elif len(numeric_cols) == 1 and num_rows == 1:
                                            num_col = numeric_cols[0]
                                            value = df[num_col].iloc[0]

                                            st.metric(
                                                label=num_col.replace('_', ' ').title(),
                                                value=f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
                                            )
                                            chart_created = True

                                        # Fallback if no chart type matched
                                        if not chart_created:
                                            st.info("📊 No suitable visualization for this data type. View the table for details.")

                                    except Exception as e:
                                        st.warning(f"Could not create visualization: {str(e)}")
                                        st.info("💡 Tip: Visualization works best with numeric data")

            else:
                st.info("👋 Start a conversation by asking a question below!")

            # Fixed input at bottom
            st.markdown("---")
            st.markdown("### 💬 Ask a Question")

            input_col1, input_col2 = st.columns([5, 1])
            with input_col1:
                user_query = st.text_input(
                    "Your question:",
                    placeholder="e.g., Which category saw the highest sales?",
                    key="user_query_input",
                    label_visibility="collapsed"
                )
            with input_col2:
                submit_button = st.button("🚀 Send", type="primary", use_container_width=True)

            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

            if submit_button and user_query:
                import time

                # Helper to update status panel in right sidebar
                def update_agent_status(agent_num, error_agent=None):
                    st.session_state.agent_status = agent_num
                    render_agent_status_native(status_placeholder, agent_num, error_agent)

                # Show processing stages
                update_agent_status(1)
                time.sleep(0.3)

                update_agent_status(2)
                time.sleep(0.2)

                # Process the actual query with error handling
                try:
                    result = orchestrator.process_query(user_query, query_type="qa")

                    update_agent_status(3)
                    time.sleep(0.2)

                    update_agent_status(4)
                    time.sleep(0.2)

                    # Show all completed
                    update_agent_status(5)
                    time.sleep(1.5)

                    # Return to ready state
                    st.session_state.agent_status = 0
                except Exception as e:
                    # Show error state on current agent
                    update_agent_status(2, error_agent=2)
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.stop()

                # Add to chat history
                st.session_state.chat_history.append({
                    'query': user_query,
                    'result': result,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                # Rerun to show new message and clear input
                st.rerun()


    else:  # Summary Mode
        st.markdown("## 📋 Generate Comprehensive Sales Summary")

        st.info("Click the button below to generate a comprehensive summary of all sales data using the multi-agent system.")

        if st.button("📊 Generate Summary", type="primary"):
            # Create two columns: main content (70%) and status panel (30%)
            col_main, col_status = st.columns([7, 3])

            with col_status:
                st.markdown("### 🔄 Agent Status")
                status_placeholder = st.empty()

            # Status update function (reuse from Q&A mode)
            def update_status(current_agent, error_agent=None):
                agents = [
                    ("🔍 Query Resolution", "Understanding summarization request"),
                    ("📊 Data Extraction", "Gathering comprehensive statistics"),
                    ("✅ Validation", "Validating data completeness"),
                    ("💬 Response Generation", "Creating executive summary")
                ]

                status_html = '''
                <div style="
                    position: sticky;
                    top: 20px;
                    padding: 16px;
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    color: white;
                ">
                    <div style="font-weight: bold; font-size: 16px; margin-bottom: 16px; text-align: center;">
                        🤖 Multi-Agent Pipeline
                    </div>
                '''

                for i, (name, description) in enumerate(agents, 1):
                    if error_agent and i == error_agent:
                        icon = '🔴'
                        status_badge = 'ERROR'
                        badge_color = '#dc3545'
                        opacity = '1.0'
                        border = 'border-left: 4px solid #dc3545;'
                    elif i < current_agent:
                        icon = '✅'
                        status_badge = 'DONE'
                        badge_color = '#28a745'
                        opacity = '0.75'
                        border = 'border-left: 4px solid #28a745;'
                    elif i == current_agent:
                        icon = '🟢'
                        status_badge = 'RUNNING'
                        badge_color = '#ffc107'
                        opacity = '1.0'
                        border = 'border-left: 4px solid #4CAF50;'
                    else:
                        icon = '⚪'
                        status_badge = 'WAITING'
                        badge_color = '#6c757d'
                        opacity = '0.5'
                        border = 'border-left: 4px solid #6c757d;'

                    status_html += f'''
                    <div style="
                        padding: 12px;
                        margin: 8px 0;
                        background: rgba(255,255,255,0.15);
                        border-radius: 8px;
                        opacity: {opacity};
                        {border}
                        transition: all 0.3s ease;
                    ">
                        <div style="display: flex; align-items: flex-start; justify-content: space-between;">
                            <div style="display: flex; align-items: flex-start; flex: 1;">
                                <span style="font-size: 20px; margin-right: 10px; margin-top: 2px;">{icon}</span>
                                <div style="flex: 1;">
                                    <div style="font-weight: 700; font-size: 14px; margin-bottom: 4px;">
                                        Agent {i}
                                    </div>
                                    <div style="font-size: 13px; font-weight: 600; margin-bottom: 3px;">
                                        {name}
                                    </div>
                                    <div style="font-size: 11px; opacity: 0.9; line-height: 1.4;">
                                        {description}
                                    </div>
                                </div>
                            </div>
                            <span style="
                                background: {badge_color};
                                padding: 4px 10px;
                                border-radius: 12px;
                                font-size: 9px;
                                font-weight: bold;
                                white-space: nowrap;
                                margin-left: 8px;
                            ">{status_badge}</span>
                        </div>
                    </div>
                    '''

                status_html += '</div>'
                status_placeholder.markdown(status_html, unsafe_allow_html=True)

            import time

            with col_main:
                update_status(1)
                time.sleep(0.3)

                update_status(2)
                time.sleep(0.2)

                try:
                    result = orchestrator.generate_summary()

                    update_status(3)
                    time.sleep(0.2)

                    update_status(4)
                    time.sleep(0.2)

                    update_status(5)
                except Exception as e:
                    update_status(2, error_agent=2)
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.stop()

                # Display summary
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 🎯 Executive Summary")
                st.markdown(result['response'])
                st.markdown('</div>', unsafe_allow_html=True)

                # Display agent workflow
                st.markdown("---")
                display_agent_workflow(result.get('metadata', {}))

                # Display visualizations
                if result.get('data'):
                    st.markdown("---")
                    display_summary_visualizations(result['data'])

                    # Display detailed statistics
                    with st.expander("📊 View Detailed Statistics"):
                        summary_stats = result['data'].get('summary_statistics', {})

                        if 'amazon_sales' in summary_stats:
                            st.markdown("#### Amazon Sales Statistics")
                            amazon_stats = summary_stats['amazon_sales']

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Orders", f"{amazon_stats.get('total_orders', 0):,}")
                            with col2:
                                st.metric("Total Revenue", f"₹{amazon_stats.get('total_revenue', 0):,.2f}")
                            with col3:
                                st.metric("Avg Order Value", f"₹{amazon_stats.get('avg_order_value', 0):,.2f}")

                        if 'international_sales' in summary_stats:
                            st.markdown("#### International Sales Statistics")
                            intl_stats = summary_stats['international_sales']

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Transactions", f"{intl_stats.get('total_transactions', 0):,}")
                            with col2:
                                st.metric("Total Revenue", f"₹{intl_stats.get('total_revenue', 0):,.2f}")
                            with col3:
                                st.metric("Unique Customers", f"{intl_stats.get('unique_customers', 0):,}")


if __name__ == "__main__":
    main()
