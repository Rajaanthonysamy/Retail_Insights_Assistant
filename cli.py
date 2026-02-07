"""
Command Line Interface for Retail Insights Assistant
Quick testing without launching the full Streamlit UI
"""
import argparse
import os
from dotenv import load_dotenv
from orchestrator import RetailInsightsOrchestrator
import json
from datetime import datetime

# Load environment variables
load_dotenv()


def print_banner():
    """Print application banner"""
    print("\n" + "=" * 70)
    print("📊 RETAIL INSIGHTS ASSISTANT - CLI Mode")
    print("=" * 70 + "\n")


def print_response(result: dict):
    """Pretty print the response"""
    print("\n" + "-" * 70)
    print("🤖 RESPONSE:")
    print("-" * 70)
    print(result['response'])
    print("-" * 70)

    # Print metadata
    print("\n📊 METADATA:")
    print(f"  Query Type: {result['query_type']}")
    print(f"  Success: {result['success']}")

    if result.get('metadata'):
        print(f"\n🔍 AGENT WORKFLOW:")
        for agent, status in result['metadata'].items():
            success = status.get('success', False)
            icon = "✅" if success else "❌"
            print(f"  {icon} {agent}: {'Completed' if success else 'Failed'}")

    if result.get('errors'):
        print(f"\n⚠️  ERRORS:")
        for error in result['errors']:
            print(f"  - {error}")

    print()


def summary_mode(orchestrator: RetailInsightsOrchestrator):
    """Run in summary mode"""
    print("\n📋 Generating comprehensive sales summary...")
    print("This may take 30-60 seconds...\n")

    start_time = datetime.now()
    result = orchestrator.generate_summary()
    end_time = datetime.now()

    elapsed = (end_time - start_time).total_seconds()

    print_response(result)
    print(f"⏱️  Processing time: {elapsed:.2f} seconds\n")


def qa_mode(orchestrator: RetailInsightsOrchestrator, question: str = None):
    """Run in Q&A mode"""
    if question:
        # Single question mode
        print(f"\n🙋 Question: {question}")
        print("Processing...\n")

        start_time = datetime.now()
        result = orchestrator.process_query(question, query_type="qa")
        end_time = datetime.now()

        elapsed = (end_time - start_time).total_seconds()

        print_response(result)
        print(f"⏱️  Processing time: {elapsed:.2f} seconds\n")

    else:
        # Interactive mode
        print("\n💬 Interactive Q&A Mode")
        print("Type your questions (or 'quit' to exit)\n")

        while True:
            try:
                question = input("❓ Your question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break

                if not question:
                    continue

                print("\nProcessing...\n")

                start_time = datetime.now()
                result = orchestrator.process_query(question, query_type="qa")
                end_time = datetime.now()

                elapsed = (end_time - start_time).total_seconds()

                print_response(result)
                print(f"⏱️  Processing time: {elapsed:.2f} seconds\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


def export_mode(orchestrator: RetailInsightsOrchestrator, output_file: str):
    """Export summary to JSON file"""
    print(f"\n📤 Exporting summary to {output_file}...")

    result = orchestrator.generate_summary()

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"✅ Summary exported successfully to {output_file}\n")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Retail Insights Assistant - CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate summary
  python cli.py --summary

  # Ask a single question
  python cli.py --question "What are the top 5 categories?"

  # Interactive Q&A mode
  python cli.py --interactive

  # Export summary to JSON
  python cli.py --export output.json

  # Use custom data path
  python cli.py --summary --data-path "path/to/data/"
        """
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Generate comprehensive sales summary'
    )

    parser.add_argument(
        '--question', '-q',
        type=str,
        help='Ask a specific question'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive Q&A session'
    )

    parser.add_argument(
        '--export', '-e',
        type=str,
        metavar='FILE',
        help='Export summary to JSON file'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='Sales Dataset/',
        help='Path to sales data directory (default: Sales Dataset/)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4-turbo-preview',
        help='OpenAI model to use (default: gpt-4-turbo-preview)'
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("   Please set it in your .env file\n")
        return 1

    # Initialize orchestrator
    try:
        print("🚀 Initializing system...")
        orchestrator = RetailInsightsOrchestrator(
            api_key=api_key,
            model_name=args.model,
            data_path=args.data_path
        )
        print("✅ System initialized successfully\n")

    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}\n")
        return 1

    # Execute based on mode
    try:
        if args.summary:
            summary_mode(orchestrator)

        elif args.question:
            qa_mode(orchestrator, args.question)

        elif args.interactive:
            qa_mode(orchestrator)

        elif args.export:
            export_mode(orchestrator, args.export)

        else:
            # No mode specified, show help
            parser.print_help()
            print("\n💡 Tip: Use --summary for a quick test or --interactive for Q&A\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        return 1

    finally:
        # Cleanup
        orchestrator.close()

    return 0


if __name__ == "__main__":
    exit(main())
