"""
LangGraph Orchestrator - Coordinates multi-agent workflow
"""
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from agents import (
    AgentState,
    QueryResolutionAgent,
    DataExtractionAgent,
    ValidationAgent,
    ResponseGenerationAgent
)
from data_processor import DataProcessor
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetailInsightsOrchestrator:
    """
    Orchestrates the multi-agent workflow using LangGraph
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo-preview", data_path: str = "Sales Dataset/"):
        """
        Initialize orchestrator with LLM and data processor

        Args:
            api_key: OpenAI API key
            model_name: Model to use for LLM agents
            data_path: Path to sales data directory
        """
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0.1
        )

        # Initialize data processor
        self.data_processor = DataProcessor(data_path)

        # Initialize agents
        self.query_agent = QueryResolutionAgent(self.llm, self.data_processor)
        self.extraction_agent = DataExtractionAgent(self.data_processor)
        self.validation_agent = ValidationAgent(self.llm)
        self.response_agent = ResponseGenerationAgent(self.llm)

        # Build workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow connecting all agents
        """
        # Create state graph
        workflow = StateGraph(AgentState)

        # Add nodes (agents) to the graph
        workflow.add_node("query_resolution", self.query_agent.run)
        workflow.add_node("data_extraction", self.extraction_agent.run)
        workflow.add_node("validation", self.validation_agent.run)
        workflow.add_node("response_generation", self.response_agent.run)

        # Define the workflow edges (sequential flow)
        workflow.set_entry_point("query_resolution")
        workflow.add_edge("query_resolution", "data_extraction")
        workflow.add_edge("data_extraction", "validation")
        workflow.add_edge("validation", "response_generation")
        workflow.add_edge("response_generation", END)

        # Compile the workflow
        return workflow.compile()

    def process_query(self, user_query: str, query_type: str = "auto") -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system

        Args:
            user_query: Natural language query from user
            query_type: 'summarization', 'qa', or 'auto' to detect automatically

        Returns:
            Dictionary containing final response and metadata
        """
        logger.info(f"Processing query: {user_query}")

        # Initialize state
        initial_state: AgentState = {
            "user_query": user_query,
            "query_type": query_type if query_type != "auto" else "",
            "structured_query": None,
            "sql_query": None,
            "extracted_data": None,
            "validation_result": None,
            "final_response": None,
            "errors": [],
            "metadata": {}
        }

        try:
            # Execute the workflow
            final_state = self.workflow.invoke(initial_state)

            # Prepare response
            response = {
                "query": user_query,
                "response": final_state.get("final_response", "No response generated"),
                "query_type": final_state.get("query_type", "unknown"),
                "data": final_state.get("extracted_data"),
                "validation": final_state.get("validation_result"),
                "metadata": final_state.get("metadata", {}),
                "errors": final_state.get("errors", []),
                "success": len(final_state.get("errors", [])) == 0
            }

            logger.info(f"Query processed successfully - Type: {response['query_type']}")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": user_query,
                "response": f"An error occurred while processing your query: {str(e)}",
                "query_type": "error",
                "data": None,
                "validation": None,
                "metadata": {},
                "errors": [str(e)],
                "success": False
            }

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of all sales data

        Returns:
            Dictionary containing summary and insights
        """
        logger.info("Generating data summary...")

        summary_query = """Generate a comprehensive summary of the overall sales performance across all datasets.
        Include key metrics like total revenue, top performing categories, regional performance, and any notable trends."""

        return self.process_query(summary_query, query_type="summarization")

    def close(self):
        """Clean up resources"""
        if self.data_processor:
            self.data_processor.close()
