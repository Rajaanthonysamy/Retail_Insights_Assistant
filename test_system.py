"""
Test script for Retail Insights Assistant
Validates core functionality of the multi-agent system
"""
import os
from dotenv import load_dotenv
from orchestrator import RetailInsightsOrchestrator
from data_processor import DataProcessor
import sys

# Load environment variables
load_dotenv()


def test_data_processor():
    """Test data processing layer"""
    print("=" * 60)
    print("TEST 1: Data Processor")
    print("=" * 60)

    try:
        processor = DataProcessor("Sales Dataset/")

        # Check loaded tables
        tables = processor.get_available_tables()
        print(f"✅ Loaded tables: {tables}")

        # Get summary statistics
        summary = processor.get_summary_statistics()
        print(f"✅ Summary statistics generated")

        for table, stats in summary.items():
            print(f"\n{table}:")
            for key, value in stats.items():
                print(f"  - {key}: {value}")

        processor.close()
        print("\n✅ Data Processor Test PASSED\n")
        return True

    except Exception as e:
        print(f"\n❌ Data Processor Test FAILED: {str(e)}\n")
        return False


def test_orchestrator_summary():
    """Test summary generation"""
    print("=" * 60)
    print("TEST 2: Summary Generation")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Please set it in your .env file")
        return False

    try:
        orchestrator = RetailInsightsOrchestrator(
            api_key=api_key,
            data_path="Sales Dataset/"
        )

        print("\n🤖 Generating summary (this may take 30-60 seconds)...")
        result = orchestrator.generate_summary()

        print("\n📊 Summary Response:")
        print("-" * 60)
        print(result['response'])
        print("-" * 60)

        # Check agent execution
        metadata = result.get('metadata', {})
        print("\n🔍 Agent Execution Status:")
        for agent, status in metadata.items():
            success = status.get('success', False)
            icon = "✅" if success else "❌"
            print(f"  {icon} {agent}: {'Success' if success else 'Failed'}")

        orchestrator.close()

        if result['success']:
            print("\n✅ Summary Generation Test PASSED\n")
            return True
        else:
            print(f"\n❌ Summary Generation Test FAILED: {result.get('errors')}\n")
            return False

    except Exception as e:
        print(f"\n❌ Summary Generation Test FAILED: {str(e)}\n")
        return False


def test_orchestrator_qa():
    """Test Q&A functionality"""
    print("=" * 60)
    print("TEST 3: Q&A Mode")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False

    try:
        orchestrator = RetailInsightsOrchestrator(
            api_key=api_key,
            data_path="Sales Dataset/"
        )

        # Test questions
        test_questions = [
            "What are the top 5 categories by sales?",
            "Which state has the highest number of orders?"
        ]

        all_passed = True

        for i, question in enumerate(test_questions, 1):
            print(f"\n🙋 Question {i}: {question}")
            print("🤖 Processing (this may take 20-30 seconds)...")

            result = orchestrator.process_query(question, query_type="qa")

            print(f"\n💬 Response:")
            print("-" * 60)
            print(result['response'])
            print("-" * 60)

            if not result['success']:
                print(f"❌ Query failed: {result.get('errors')}")
                all_passed = False
            else:
                print(f"✅ Query successful")

        orchestrator.close()

        if all_passed:
            print("\n✅ Q&A Mode Test PASSED\n")
            return True
        else:
            print("\n❌ Q&A Mode Test FAILED\n")
            return False

    except Exception as e:
        print(f"\n❌ Q&A Mode Test FAILED: {str(e)}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RETAIL INSIGHTS ASSISTANT - SYSTEM TESTS")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Data Processor
    results.append(("Data Processor", test_data_processor()))

    # Test 2: Summary Generation (requires API key)
    if os.getenv("OPENAI_API_KEY"):
        results.append(("Summary Generation", test_orchestrator_summary()))
        results.append(("Q&A Mode", test_orchestrator_qa()))
    else:
        print("\n⚠️  OPENAI_API_KEY not found. Skipping LLM-based tests.")
        print("   Set OPENAI_API_KEY in .env to run full tests.\n")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        icon = "✅" if passed else "❌"
        status = "PASSED" if passed else "FAILED"
        print(f"{icon} {test_name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Please review errors above")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
