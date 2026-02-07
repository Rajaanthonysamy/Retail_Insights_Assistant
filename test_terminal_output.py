"""
Test to verify SQL queries are printed to terminal
"""
import os
from dotenv import load_dotenv
from orchestrator import RetailInsightsOrchestrator

load_dotenv()

# Initialize orchestrator
orchestrator = RetailInsightsOrchestrator(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("MODEL_NAME", "gpt-4-turbo-preview"),
    data_path="Sales Dataset/"
)

# Test queries
test_queries = [
    "What are the top 5 categories by revenue in Q2 2022?",
    "Which state had the highest sales in April 2022?",
    "Show me the revenue breakdown by category"
]

print("\n" + "🎯" * 40)
print("TESTING SQL QUERY TERMINAL OUTPUT")
print("🎯" * 40 + "\n")

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"TEST #{i}: {query}")
    print("="*80)

    result = orchestrator.process_query(query, query_type="qa")

    print(f"\n✅ Response received (showing first 200 chars):")
    print(result['response'][:200] + "...\n")

print("\n" + "🎯" * 40)
print("✅ All tests completed!")
print("🎯" * 40 + "\n")
