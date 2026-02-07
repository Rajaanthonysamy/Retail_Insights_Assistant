# 📊 Retail Insights Assistant

A multi-agent GenAI-powered system for analyzing retail sales data, generating automated business insights, and answering analytical questions in natural language.

## 🎯 Overview

This project implements an intelligent **Retail Insights Assistant** that combines:
- **Multi-Agent Architecture**: Using LangGraph with 4 specialized agents
- **Efficient Data Processing**: DuckDB for high-performance SQL queries
- **LLM Integration**: OpenAI GPT-4 for natural language understanding
- **Interactive UI**: Streamlit-based interface for easy interaction
- **Scalable Design**: Architecture ready for 100GB+ datasets

## 🏗️ Architecture

### Multi-Agent System

The system uses **LangGraph** to orchestrate 4 specialized agents:

1. **Query Resolution Agent**
   - Interprets natural language queries
   - Converts user intent to structured queries
   - Generates optimized SQL queries for DuckDB

2. **Data Extraction Agent**
   - Executes SQL queries against DuckDB
   - Retrieves relevant data efficiently
   - Handles both summarization and specific Q&A queries

3. **Validation Agent**
   - Validates extracted data quality
   - Checks for inconsistencies and anomalies
   - Provides confidence scores and recommendations

4. **Response Generation Agent**
   - Generates human-readable insights
   - Formats responses for business users
   - Includes specific metrics and actionable recommendations

### Technology Stack

- **Language**: Python 3.8+
- **LLM Framework**: LangChain, LangGraph, OpenAI GPT-4
- **Data Processing**: DuckDB, Pandas
- **UI**: Streamlit
- **Vector Store** (optional): ChromaDB, FAISS
- **Orchestration**: LangGraph state machine

## 📋 Features

### ✅ Implemented Features

- ✅ Multi-agent system with 4 specialized agents
- ✅ Two operational modes:
  - **Summarization Mode**: Comprehensive sales performance summaries
  - **Conversational Q&A Mode**: Ad-hoc analytical questions
- ✅ DuckDB integration for efficient querying
- ✅ Streamlit UI with interactive visualizations
- ✅ Prompt engineering for consistent responses
- ✅ Conversation history and context management
- ✅ Agent workflow visualization
- ✅ Real-time data visualizations (charts and graphs)

### 🎯 Use Cases

1. **Executive Summaries**: Generate comprehensive performance reports
2. **Ad-hoc Analysis**: Answer specific business questions
3. **Trend Analysis**: Identify patterns in sales data
4. **Regional Performance**: Analyze sales by geography
5. **Category Insights**: Understand product category performance

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- pip (Python package manager)

### Installation Steps

1. **Clone or extract the project**
   ```bash
   cd blend_assignment
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Verify data files**
   Ensure the `Sales Dataset/` folder contains:
   - Amazon Sale Report.csv
   - Sale Report.csv
   - International sale Report.csv

### Running the Application

**Start the Streamlit UI:**
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the CLI (Optional)

For programmatic access, you can use the orchestrator directly:

```python
from orchestrator import RetailInsightsOrchestrator
import os

# Initialize
orchestrator = RetailInsightsOrchestrator(
    api_key=os.getenv("OPENAI_API_KEY"),
    data_path="Sales Dataset/"
)

# Generate summary
summary = orchestrator.generate_summary()
print(summary['response'])

# Ask a question
result = orchestrator.process_query(
    "Which category has the highest sales?"
)
print(result['response'])
```

## 💡 Usage Examples

### Example Questions (Q&A Mode)

- "Which category saw the highest sales in April 2022?"
- "What is the total revenue by state?"
- "Show me the top 10 performing products"
- "Which region has the most orders?"
- "What is the average order value?"
- "Which customers made the most purchases?"
- "What are the sales trends by month?"

### Summary Mode

Click "Generate Summary" to get a comprehensive overview including:
- Total revenue and order metrics
- Top performing categories
- Regional performance analysis
- Key trends and insights

## 🗂️ Project Structure

```
blend_assignment/
├── app.py                      # Streamlit UI application
├── orchestrator.py             # LangGraph workflow orchestrator
├── agents.py                   # Multi-agent system implementation
├── data_processor.py           # DuckDB data processing layer
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
├── ARCHITECTURE.md            # 100GB+ scalability architecture
└── Sales Dataset/             # Sales data CSV files
    ├── Amazon Sale Report.csv
    ├── Sale Report.csv
    └── International sale Report.csv
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `MODEL_NAME`: OpenAI model to use (default: gpt-4-turbo-preview)
- `TEMPERATURE`: LLM temperature (default: 0.1)
- `DATA_PATH`: Path to sales data (default: Sales Dataset/)

### Model Configuration

You can modify the model in `config.py`:
- **GPT-4 Turbo**: Best performance, higher cost
- **GPT-3.5 Turbo**: Faster, lower cost (change in .env)

## 📊 Data Processing

### DuckDB Integration

- **In-memory database** for fast queries
- **Automatic CSV loading** on initialization
- **Optimized SQL queries** for analytics
- **Supports aggregations** and complex joins

### Supported Datasets

1. **Amazon Sales**: Order-level transaction data
2. **Inventory**: Stock and SKU information
3. **International Sales**: Customer transaction data

## 🧪 Testing

### Manual Testing

1. Start the application
2. Try example questions in Q&A mode
3. Generate summary in Summary mode
4. Verify agent workflow execution
5. Check data visualizations

### Expected Behavior

- Agents should complete successfully (green checkmarks)
- Responses should be contextual and accurate
- Visualizations should display correctly
- Chat history should persist during session

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   - Solution: Ensure `OPENAI_API_KEY` is set in `.env`

2. **Data Loading Error**
   - Solution: Verify CSV files exist in `Sales Dataset/` folder

3. **Module Not Found**
   - Solution: Run `pip install -r requirements.txt`

4. **Slow Performance**
   - Solution: Consider using GPT-3.5-turbo for faster responses

## 🎨 UI Features

- **Dual Mode Interface**: Switch between Q&A and Summary modes
- **Agent Workflow Visualization**: See each agent's status
- **Interactive Charts**: Plotly-based visualizations
- **Chat History**: Persistent conversation history
- **Responsive Design**: Works on different screen sizes

## 🔐 Security & Privacy

- API keys stored in environment variables (not in code)
- No data sent to external services except OpenAI API
- Local data processing with DuckDB
- No persistent storage of conversations

## 📈 Performance Considerations

- **Current Scale**: Optimized for datasets up to 10GB
- **In-memory Processing**: Fast queries with DuckDB
- **LLM Caching**: Reduces API calls for similar queries
- **Async Processing**: Non-blocking UI operations

For 100GB+ scaling strategy, see **ARCHITECTURE.md**

## 🔮 Future Enhancements

### Potential Improvements

1. **Vector Embeddings**: Add FAISS for semantic search
2. **Query Caching**: Cache common query results
3. **Batch Processing**: Support bulk data analysis
4. **Export Features**: Download reports as PDF/Excel
5. **Authentication**: Add user management
6. **Real-time Data**: Streaming data ingestion
7. **Multi-language**: Support for non-English queries

## 🤝 Assumptions & Limitations

### Assumptions

- CSV files are properly formatted
- Dates are in consistent format
- Currency is in INR (Indian Rupees)
- Data quality is reasonable (minimal nulls/errors)

### Current Limitations

1. **API Dependency**: Requires OpenAI API access
2. **Memory Constraints**: In-memory DuckDB limited by RAM
3. **No Persistence**: Chat history not saved between sessions
4. **Limited Time-series**: No advanced forecasting
5. **Single User**: Not designed for concurrent users

## 📚 Technical Notes

### Agent Communication

Agents communicate through a shared **AgentState** object:
```python
class AgentState(TypedDict):
    user_query: str
    query_type: str
    structured_query: Optional[str]
    sql_query: Optional[str]
    extracted_data: Optional[Dict]
    validation_result: Optional[Dict]
    final_response: Optional[str]
    errors: List[str]
    metadata: Dict
```

### LangGraph Workflow

The workflow follows a linear pipeline:
```
Query Resolution → Data Extraction → Validation → Response Generation
```

Each agent can:
- Read the current state
- Modify relevant fields
- Add errors or metadata
- Pass control to the next agent

### Prompt Engineering

Key prompt strategies used:
- **System prompts** for consistent behavior
- **Few-shot examples** (implicit in prompts)
- **Structured output** (JSON formatting)
- **Context preservation** across agents
- **Temperature control** for consistency

## 📄 License

This project is created for the Blend360 GenAI Interview Assignment.

## 👥 Contact

For questions or issues, please refer to the assignment submission guidelines.

---

**Built with** ❤️ **using LangGraph, OpenAI GPT-4, DuckDB, and Streamlit**
