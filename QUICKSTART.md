# 🚀 Quick Start Guide

Get up and running with the Retail Insights Assistant in 5 minutes!

## Prerequisites

- ✅ Python 3.8+ installed
- ✅ OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- ✅ Terminal/Command Prompt access

## Step-by-Step Setup

### 1. Navigate to Project Directory
```bash
cd blend_assignment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed langchain-0.1.0 streamlit-1.29.0 duckdb-0.9.2 ...
```

### 3. Configure API Key

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-api-key-here
```

**💡 Tip:** Get your API key from https://platform.openai.com/api-keys

### 4. Verify Setup (Optional)
```bash
python test_system.py
```

This will:
- ✅ Verify data files are loaded
- ✅ Test data processing
- ✅ Test multi-agent system (if API key is set)

### 5. Launch the Application
```bash
streamlit run app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

The app will automatically open in your browser! 🎉

---

## Using the Application

### Mode 1: Generate Summary 📋

1. Click **"📋 Generate Summary"** in the sidebar
2. Click **"📊 Generate Summary"** button
3. Wait ~30 seconds for the multi-agent system to analyze all data
4. View comprehensive insights with charts

**What you'll see:**
- Executive summary
- Top categories performance
- Regional analysis
- Agent workflow status
- Interactive visualizations

### Mode 2: Ask Questions 💬

1. Select **"💬 Conversational Q&A"** in the sidebar
2. Type a question like:
   - "Which category has the highest sales?"
   - "What is the total revenue by state?"
   - "Show me top 10 performing products"
3. Click **"🚀 Ask"**
4. Wait ~20 seconds for the response

**Features:**
- Chat history (persists during session)
- Agent workflow visualization
- Data tables for query results
- Clear history option

---

## Example Questions to Try

### Sales Performance
- "What are the top 5 categories by revenue?"
- "Which category had the most orders?"
- "Show me sales by category"

### Regional Analysis
- "Which state has the highest sales?"
- "What is the total revenue by region?"
- "Which cities have the most orders?"

### Customer Insights
- "Who are the top customers by purchase value?"
- "How many unique customers do we have?"

### Trends
- "What are the monthly sales trends?"
- "Which products are most popular?"

---

## Troubleshooting

### Issue: "OpenAI API key not found"
**Solution:**
- Ensure `.env` file exists in project root
- Verify `OPENAI_API_KEY=sk-...` is set correctly
- Restart the Streamlit app

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Data files not found"
**Solution:**
- Verify `Sales Dataset/` folder exists
- Check that CSV files are present:
  ```bash
  ls "Sales Dataset/"
  ```

### Issue: "Port 8501 already in use"
**Solution:**
```bash
streamlit run app.py --server.port=8502
```

---

## What's Happening Behind the Scenes?

When you ask a question, the **multi-agent system**:

1. **Query Resolution Agent** 🧠
   - Understands your natural language question
   - Converts it to a structured query
   - Generates optimized SQL

2. **Data Extraction Agent** 📊
   - Executes SQL query on DuckDB
   - Retrieves relevant data
   - Formats results

3. **Validation Agent** ✅
   - Checks data quality
   - Validates results
   - Provides confidence score

4. **Response Generation Agent** 💬
   - Creates human-readable insights
   - Includes specific metrics
   - Formats professionally

All of this happens in **15-30 seconds**!

---

## Next Steps

- 📖 Read the full [README.md](README.md) for detailed documentation
- 🏗️ Review [ARCHITECTURE.md](ARCHITECTURE.md) for scalability design
- 🧪 Run `python test_system.py` to validate functionality
- 🎨 Customize prompts in `agents.py` for your use case

---

## Need Help?

- Check [README.md](README.md) for comprehensive documentation
- Review error messages in the terminal
- Ensure all CSV files are in the correct format
- Verify your OpenAI API key is valid and has credits

---

**Happy Analyzing! 📊✨**
