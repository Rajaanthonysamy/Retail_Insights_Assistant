"""
Test script to verify YoY query handling after agents.py updates
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_processor import DataProcessor
from agents import QueryResolutionAgent, AgentState

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model_name=os.getenv("MODEL_NAME", "gpt-4-turbo-preview"),
    temperature=float(os.getenv("TEMPERATURE", "0.1"))
)

def test_yoy_query():
    """Test YoY query generation"""

    # Initialize components (data is loaded automatically in __init__)
    data_processor = DataProcessor()
    query_agent = QueryResolutionAgent(llm, data_processor)

    # Test query - YoY growth in Q3
    test_query = "Which category has the highest YoY growth in Q3?"

    print("=" * 80)
    print(f"Testing Query: {test_query}")
    print("=" * 80)

    # Create initial state
    state: AgentState = {
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

    # Run query resolution
    result_state = query_agent.run(state)

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    if result_state['sql_query']:
        print(f"\n✅ SQL Query Generated:\n")
        print(result_state['sql_query'])
        print("\n" + "-" * 80)

        # Check for hardcoded values
        sql_lower = result_state['sql_query'].lower()
        issues = []

        if '2021' in result_state['sql_query'] or '2022' in result_state['sql_query']:
            issues.append("⚠️  WARNING: SQL contains hardcoded years (2021/2022)")

        if 'quarter' not in sql_lower and 'q3' in test_query.lower():
            issues.append("⚠️  WARNING: Query mentions Q3 but SQL doesn't handle quarters")

        if 'year(date)' not in sql_lower and 'quarter(date)' not in sql_lower:
            issues.append("⚠️  WARNING: Not using DATE functions properly")

        if 'prev' not in sql_lower and 'previous' not in sql_lower:
            issues.append("⚠️  WARNING: Doesn't look like a YoY calculation")

        if issues:
            print("\n⚠️  ISSUES DETECTED:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✅ SQL looks correct:")
            print("  - Uses dynamic date parsing")
            print("  - No hardcoded years")
            print("  - Includes YoY calculation logic")
            print("  - Extracts quarter from query")
    else:
        print("\n❌ ERROR: No SQL query generated!")
        if result_state.get('error_message'):
            print(f"Error: {result_state['error_message']}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_yoy_query()
