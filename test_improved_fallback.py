"""
Test improved fallback messages for queries that return empty results
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_processor import DataProcessor
from agents import QueryResolutionAgent, DataExtractionAgent, ResponseGenerationAgent

load_dotenv()

# Initialize components
llm = ChatOpenAI(
    model_name=os.getenv("MODEL_NAME", "gpt-4-turbo-preview"),
    temperature=float(os.getenv("TEMPERATURE", "0.1"))
)

data_processor = DataProcessor()
query_agent = QueryResolutionAgent(llm, data_processor)
extraction_agent = DataExtractionAgent(data_processor)
response_agent = ResponseGenerationAgent(llm)

# Test with YoY Q3 query
test_query = "Which category has the highest YoY growth in Q3 in the North region?"

print("=" * 80)
print(f"Testing Query: {test_query}")
print("=" * 80)

# Step 1: Query Resolution
state = {
    "user_query": test_query,
    "query_type": "qa",
    "structured_query": None,
    "sql_query": None,
    "extracted_data": None,
    "validation_result": None,
    "final_response": None,
    "errors": [],
    "metadata": {}
}

print("\n📝 Step 1: Query Resolution")
print("-" * 80)
state = query_agent.run(state)
print(f"SQL Generated: {state['sql_query'][:100]}...")

# Step 2: Data Extraction
print("\n📊 Step 2: Data Extraction")
print("-" * 80)
state = extraction_agent.run(state)
extracted = state.get('extracted_data', {})
print(f"Row Count: {extracted.get('row_count', 0)}")
print(f"Empty Result: {extracted.get('empty_result', False)}")

if extracted.get('dataset_metadata'):
    metadata = extracted['dataset_metadata']
    print(f"\nDataset Metadata:")
    if 'date_range' in metadata:
        dr = metadata['date_range']
        print(f"  Date Range: {dr.get('min_date')} to {dr.get('max_date')}")
        print(f"  Available Years: {dr.get('available_years')}")
        print(f"  Available Quarters: {dr.get('available_quarters')}")
        print(f"  Unique Years: {dr.get('unique_years')}")

# Step 3: Response Generation
print("\n💬 Step 3: Response Generation")
print("-" * 80)
state = response_agent.run(state)

print("\n📋 FINAL RESPONSE:")
print("=" * 80)
print(state['final_response'])
print("=" * 80)
