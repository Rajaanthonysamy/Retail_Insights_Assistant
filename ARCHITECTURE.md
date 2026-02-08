# Retail Insights Assistant - Architecture Document

## Executive Summary

The Retail Insights Assistant is a **Multi-Agent GenAI System** for sales analytics. It uses a 4-agent LangGraph pipeline powered by OpenAI GPT-4 to convert natural language questions into SQL queries, extract data from DuckDB, validate results, and generate business-friendly insights. The system features a Streamlit web interface with real-time agent status monitoring, interactive visualizations, and cloud-ready deployment.

---

## System Architecture Overview

```
                              User (Browser)
                                   |
                          +--------v--------+
                          |   Streamlit UI   |
                          |    (app.py)      |
                          +--------+---------+
                                   |
                    +--------------v--------------+
                    |   RetailInsightsOrchestrator |
                    |      (orchestrator.py)       |
                    +--------------+--------------+
                                   |
                          LangGraph StateGraph
                                   |
        +----------+----------+----------+----------+
        |          |          |          |          |
        v          v          v          v          |
   +---------+ +---------+ +---------+ +---------+ |
   | Agent 1 | | Agent 2 | | Agent 3 | | Agent 4 | |
   | Query   | | Data    | | Valid-  | | Response| |
   | Resoln. | | Extract | | ation   | | Gen.    | |
   +---------+ +---------+ +---------+ +---------+ |
        |          |                                |
        v          v                                |
   +---------+ +---------+                          |
   | OpenAI  | | DuckDB  |                          |
   | GPT-4   | | Engine  |                          |
   +---------+ +---------+                          |
                   |                                |
              +----v----+                           |
              | CSV Data |                          |
              | (3 files)|                          |
              +----------+                          |
```

---

## Component Architecture

### 1. Streamlit UI Layer (`app.py` - 884 lines)

The web interface built with Streamlit providing:

**Layout Structure:**
- **Left Sidebar (st.sidebar):**
  - API Key input (password field, session-only storage)
  - Mode selection: Conversational Q&A / Generate Summary
  - Example questions for user guidance
- **Main Content Area (70% width):**
  - Chat interface with message history (Q&A mode)
  - Executive summary with metrics (Summary mode)
  - Expandable details: SQL query, data tables, interactive charts
- **Right Panel (30% width, column-based):**
  - Real-time Agent Pipeline status
  - Color-coded agent status badges (Ready/Running/Done/Waiting/Error)
  - Collapsible agent detail cards

**Key Features:**
- `get_api_key()`: Retrieves API key from `st.session_state` only (not `.env`)
- `@st.cache_resource` on `initialize_orchestrator()`: Caches orchestrator per API key
- `render_agent_status_native()`: Live agent status with HTML/CSS rendering
- Smart Visualization Engine: Auto-selects chart type based on data characteristics
  - Time series data -> Line chart
  - 2-8 categories -> Donut chart
  - 8+ categories -> Horizontal bar chart
  - Single value -> Metric card
  - Multiple metrics -> Grouped bar chart
  - Two numeric columns -> Scatter plot
- `st.columns([7, 3])` layout for main content + agent sidebar

**State Management:**
- `st.session_state.chat_history`: Persists chat messages within session
- `st.session_state.agent_status`: Tracks current agent (0=ready, 1-4=processing)
- `st.session_state.openai_api_key`: Session-only API key storage

---

### 2. LangGraph Orchestrator (`orchestrator.py` - 166 lines)

**Class: `RetailInsightsOrchestrator`**

Coordinates the multi-agent workflow using LangGraph's StateGraph.

**Initialization:**
```python
RetailInsightsOrchestrator(
    api_key: str,           # OpenAI API key (from UI)
    model_name: str,        # Default: "gpt-4-turbo-preview"
    data_path: str          # Default: "Sales Dataset/"
)
```

**Components Initialized:**
- `ChatOpenAI` LLM instance (temperature=0.1)
- `DataProcessor` for DuckDB operations
- 4 Agent instances (Query, Extraction, Validation, Response)
- Compiled LangGraph workflow

**Workflow Graph:**
```
Entry -> query_resolution -> data_extraction -> validation -> response_generation -> END
```

- Sequential flow (no branching) for reliability
- Each node is an agent's `run()` method
- State passed via `AgentState` TypedDict

**Public Methods:**
- `process_query(user_query, query_type)`: Process a single NL query
- `generate_summary()`: Generate comprehensive summary across all datasets
- `close()`: Clean up resources

**Response Format:**
```python
{
    "query": str,           # Original user query
    "response": str,        # Generated natural language response
    "query_type": str,      # "qa" | "summarization" | "error"
    "sql_query": str,       # Generated SQL (logged to terminal)
    "data": dict,           # Extracted data (tables, charts)
    "validation": dict,     # Validation results + confidence
    "metadata": dict,       # Per-agent success/failure tracking
    "errors": list,         # Error messages from pipeline
    "success": bool         # Overall success flag
}
```

---

### 3. Agent System (`agents.py` - 541 lines)

**Shared State: `AgentState` (TypedDict)**
```python
class AgentState(TypedDict):
    user_query: str
    query_type: str                        # "summarization" | "qa"
    structured_query: Optional[str]
    sql_query: Optional[str]
    extracted_data: Optional[Dict]
    validation_result: Optional[Dict]
    final_response: Optional[str]
    errors: Annotated[List[str], operator.add]  # Accumulated errors
    metadata: Dict[str, Any]                     # Per-agent metadata
```

#### Agent 1: QueryResolutionAgent
**Purpose:** Converts natural language to SQL

**Process:**
1. Retrieves table schemas, sample data, and value examples via `DataProcessor`
2. Constructs a rich system prompt with:
   - Full database context (schemas, row counts, sample values)
   - DuckDB-specific SQL rules (column quoting, date functions)
   - YoY/growth rate query patterns with dynamic parameter extraction
   - Common query templates
3. Sends to GPT-4 for NL-to-SQL conversion
4. Parses JSON response: `{query_type, structured_query, sql_query, required_tables}`
5. Handles JSON parse failures with a simple fallback

**Key Design:**
- Date column is DATE type (DuckDB auto-detection)
- Uses `YEAR()`, `QUARTER()`, `MONTH()` functions directly
- Dynamic YoY patterns using self-joins (no hardcoded years)
- Context truncated to 3500 chars to stay within token limits

#### Agent 2: DataExtractionAgent
**Purpose:** Executes SQL queries and retrieves data

**Process:**
1. For **summarization**: Gathers summary stats, top categories, regional performance
2. For **Q&A with SQL**: Executes SQL query via DuckDB
3. For **empty results**: Adds fallback data + dataset metadata for context
4. For **SQL errors**: Provides smart fallback based on query intent:
   - Category queries -> `get_top_categories(10)`
   - Region/state queries -> `get_regional_performance()`
   - Other queries -> `get_summary_statistics()`
5. For **no SQL generated**: Same smart fallback strategy

**Fallback Logic:** Three-tier fallback ensures the system always returns useful data.

#### Agent 3: ValidationAgent
**Purpose:** Validates extracted data quality

**Process:**
1. Checks if data was extracted successfully
2. Validates data completeness (summarization: checks required keys)
3. Checks for empty Q&A results (reduces confidence)
4. Uses LLM for semantic validation:
   - Is data relevant to the query?
   - Any anomalies or inconsistencies?
   - Sufficient quality to answer?
5. Assigns confidence score (0.0 - 1.0)

**Output:**
```python
{
    "is_valid": bool,
    "confidence": float,      # 0.0 to 1.0
    "issues": [str],
    "recommendations": [str],
    "llm_assessment": str      # LLM's semantic review
}
```

#### Agent 4: ResponseGenerationAgent
**Purpose:** Generates business-friendly natural language responses

**Process:**
1. Analyzes data limitations (empty results, fallbacks, errors)
2. For empty results: Explains WHY using dataset metadata
   - Checks for YoY queries with insufficient years
   - Checks for quarter availability (Q3/Q4 may not exist)
3. Constructs prompt with: data to present, validation status, fallback notes
4. GPT-4 generates executive-style response with:
   - Direct answer to user's question
   - Specific numbers and percentages
   - Key trends and patterns
   - Business-friendly language
   - Data limitation explanations when applicable

---

### 4. Data Processing Layer (`data_processor.py` - 317 lines)

**Class: `DataProcessor`**

**Initialization:**
- Connects to DuckDB (persistent file: `retail_data.duckdb`)
- Auto-loads CSV files into tables if tables don't exist
- Checks `information_schema.tables` to avoid re-loading

**Three Datasets:**

| Table Name | Source File | Key Columns |
|---|---|---|
| `amazon_sales` | Amazon Sale Report.csv | Order ID, Category, Status, Amount, Date, ship-state |
| `inventory` | Sale Report.csv | Category, Size, Color, Stock |
| `international_sales` | International sale Report.csv | CUSTOMER, Months, PCS, GROSS AMT |

**Key Methods:**
- `execute_query(sql)`: Execute SQL, return DataFrame
- `get_table_schema(table)`: DESCRIBE table
- `get_table_context(table)`: Schema + sample data + stats + value examples (for LLM prompts)
- `get_summary_statistics()`: Cross-dataset summary (revenue, orders, customers)
- `get_dataset_metadata()`: Date ranges, available years/quarters
- `get_top_categories(limit)`: Top categories by revenue
- `get_regional_performance()`: Sales by state

**Cloud Deployment Handling:**
- DuckDB file is ephemeral (`.gitignore` includes `*.duckdb`)
- CSV files are committed to git
- On cold start: `_load_datasets()` creates tables from CSVs
- Subsequent requests reuse existing tables in the database

---

## Technology Stack

| Technology | Version | Purpose |
|---|---|---|
| **LangGraph** | 1.0.1 | Agent workflow orchestration (StateGraph) |
| **OpenAI GPT-4** | via openai 2.9.0 | NL understanding, SQL generation, insights |
| **LangChain** | 0.3.27 | ChatOpenAI, message abstractions |
| **LangChain-Core** | 0.3.80 | HumanMessage, SystemMessage |
| **LangChain-OpenAI** | 0.3.35 | OpenAI integration |
| **DuckDB** | 1.4.4 | Columnar OLAP analytics engine |
| **Streamlit** | 1.49.1 | Web UI framework |
| **Plotly** | 6.5.2 | Interactive visualizations |
| **Pandas** | 2.2.3 | Data manipulation |
| **NumPy** | 2.2.2 | Numerical computing |
| **Pydantic** | 2.11.7 | Data validation |
| **tiktoken** | 0.9.0 | Token counting |
| **python-dotenv** | 1.0.1 | Environment variable management |

---

## Data Flow

### Q&A Mode
```
1. User types question in chat input
2. app.py updates agent status to "Agent 1: Running"
3. QueryResolutionAgent:
   - Fetches table contexts from DuckDB
   - Sends NL query + schema to GPT-4
   - Receives structured JSON with SQL query
4. app.py updates agent status to "Agent 2: Running"
5. DataExtractionAgent:
   - Executes SQL on DuckDB
   - If error/empty: provides fallback data
   - Attaches dataset metadata
6. app.py updates agent status to "Agent 3: Running"
7. ValidationAgent:
   - Validates data completeness & relevance
   - LLM semantic validation
   - Assigns confidence score
8. app.py updates agent status to "Agent 4: Running"
9. ResponseGenerationAgent:
   - Analyzes data limitations
   - Generates business-friendly response via GPT-4
10. app.py shows "All Done", then resets to "Ready"
11. Response displayed in chat with expandable details:
    - Generated SQL query
    - Agent workflow status
    - Data table (with DataFrame viewer)
    - Interactive chart (auto-selected type)
```

### Summary Mode
```
1. User clicks "Generate Summary" button
2. Orchestrator sends pre-defined summary query
3. DataExtractionAgent gathers:
   - Summary statistics (all 3 tables)
   - Top 10 categories by revenue
   - Top 10 states by revenue
4. ResponseGenerationAgent creates executive summary
5. UI displays:
   - Executive summary text
   - Agent workflow status
   - Top Categories bar chart
   - Regional Performance bar chart
   - Detailed statistics in expanders
```

---

## Error Handling & Resilience

### Multi-Level Fallback Strategy
```
Level 1: SQL query succeeds -> return exact results
Level 2: SQL fails -> smart fallback based on query intent
Level 3: No SQL generated -> provide relevant summary data
Level 4: Agent error -> propagate error, continue pipeline
Level 5: Complete failure -> return error message with context
```

### Per-Agent Error Isolation
- Each agent wrapped in try/except
- Errors appended to `state['errors']` list (accumulated via `operator.add`)
- Pipeline continues even if individual agents fail
- Validation agent assigns lower confidence on partial data

### UI Error Feedback
- Agent status shows error state (red badge) on failure
- Error messages displayed to user
- Chat continues to work after errors

---

## Security

- **API Key Management:**
  - User enters key via Streamlit UI (password field)
  - Key stored in `st.session_state` only (memory)
  - Never persisted to disk or `.env`
  - Masked display: `sk-a...1234`
  - "Change API Key" clears cached orchestrator
- **Data Security:**
  - DuckDB queries are parameterized where possible
  - No external data persistence
  - Session-scoped chat history

---

## Deployment

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
# Enter API key in browser sidebar
```

### Streamlit Cloud
1. Push code to GitHub (CSV files included in repo)
2. Connect repo on share.streamlit.io
3. DuckDB auto-rebuilds from CSV on cold start
4. `.duckdb` files in `.gitignore` (ephemeral filesystem)
5. `requirements.txt` has pinned versions for reproducibility
6. API key entered by user at runtime (no secrets file needed)

### Docker (Production)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## Project Structure

```
blend_assignment/
  app.py                  # Streamlit UI (884 lines)
  orchestrator.py         # LangGraph workflow (166 lines)
  agents.py               # 4 Agent classes (541 lines)
  data_processor.py       # DuckDB data layer (317 lines)
  requirements.txt        # Pinned dependencies (26 packages)
  .env                    # Environment variables (optional)
  .gitignore              # Excludes .duckdb, __pycache__, .env
  ARCHITECTURE.md         # This file
  Sales Dataset/
    Amazon Sale Report.csv
    Sale Report.csv
    International sale Report.csv
    Cloud Warehouse Compersion Chart.csv
    Expense IIGF.csv
    May-2022.csv
    P  L March 2021.csv
```

---

## Scalability Roadmap (100GB+)

For scaling beyond the current implementation, see the following considerations:

### Data Layer
- **Current:** DuckDB (embedded, single-machine)
- **Scale:** BigQuery / Snowflake / Databricks SQL
- **Strategy:** Partition by year/quarter, use Delta Lake/Parquet

### Processing
- **Current:** Pandas + DuckDB
- **Scale:** PySpark / Dask for distributed processing
- **Strategy:** Apache Airflow for ETL orchestration

### LLM Optimization
- **Current:** Direct OpenAI API calls
- **Scale:** Redis caching, prompt optimization, model routing
- **Strategy:** GPT-3.5 for simple queries, GPT-4 for complex

### Retrieval Enhancement
- **Current:** SQL-based query execution
- **Scale:** RAG with vector embeddings (FAISS/Pinecone)
- **Strategy:** Semantic search over data partition summaries

### Infrastructure
- **Current:** Single Streamlit instance
- **Scale:** Kubernetes with horizontal pod autoscaling
- **Strategy:** Docker containers, load balancer, Redis cache

### Monitoring
- **Current:** Python logging
- **Scale:** Prometheus + Grafana, structured logging, alerting
- **Strategy:** Track query latency, error rates, LLM costs, cache hit rates

### Estimated Costs (100GB, 10K queries/month)
- Data Storage (S3 + BigQuery): $150-300/month
- LLM API Costs: $500-1,500/month
- Compute (K8s cluster): $200-500/month
- **Total: ~$1,000-2,500/month**
