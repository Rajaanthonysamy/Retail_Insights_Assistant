"""
Multi-Agent System using LangGraph
Implements: Query Resolution Agent, Data Extraction Agent, Validation Agent
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import operator
import logging
from data_processor import DataProcessor
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the state that will be passed between agents
class AgentState(TypedDict):
    """State shared across all agents"""
    user_query: str
    query_type: str  # 'summarization' or 'qa'
    structured_query: Optional[str]
    sql_query: Optional[str]
    extracted_data: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    final_response: Optional[str]
    errors: Annotated[List[str], operator.add]
    metadata: Dict[str, Any]


class QueryResolutionAgent:
    """
    Agent 1: Converts natural language queries into structured queries
    """

    def __init__(self, llm: ChatOpenAI, data_processor: DataProcessor):
        self.llm = llm
        self.data_processor = data_processor
        self.name = "QueryResolutionAgent"

    def run(self, state: AgentState) -> AgentState:
        """
        Analyze user query and convert to structured format
        """
        logger.info(f"[{self.name}] Processing user query...")

        try:
            # Get available tables and their comprehensive context
            tables = self.data_processor.get_available_tables()
            table_contexts = {
                table: self.data_processor.get_table_context(table)
                for table in tables
            }

            logger.info(f"[{self.name}] Retrieved table contexts for query resolution")
            logger.info(f"[{self.name}] Available tables: {tables}")

            # Create prompt for query resolution with rich context
            system_prompt = f"""You are a Query Resolution Agent for a retail sales analytics system.

Your task is to analyze the user's natural language query and convert it into a valid SQL query.

DATABASE CONTEXT (Tables with Schemas, Sample Data & Statistics):
{json.dumps(table_contexts, indent=2, default=str)[:3500]}

KEY INSIGHTS:
- amazon_sales table has {table_contexts.get('amazon_sales', {}).get('statistics', {}).get('total_rows', 0):,} records
- Available Categories: {table_contexts.get('amazon_sales', {}).get('value_examples', {}).get('Category', [])}
- Date format: MM-DD-YY stored as STRING (e.g., "04-30-22")

CRITICAL SQL Rules:
1. ALWAYS use proper GROUP BY clauses - every non-aggregated column in SELECT must be in GROUP BY
2. For DuckDB, use standard SQL syntax (no Oracle-specific functions)
3. Column names with spaces or special characters must be quoted with double quotes: "Order ID", "ship-state"
4. ALWAYS filter out cancelled orders: WHERE Status != 'Cancelled' AND Amount > 0
5. Column names are case-sensitive - use exact names from schema

DATE HANDLING (CRITICAL - Date column is DATE type in DuckDB):
- Date column in amazon_sales is already DATE type (auto-detected by DuckDB)
- Extract year: YEAR(Date)
- Extract quarter: QUARTER(Date)
- Extract month: MONTH(Date)
- Quarter mapping: Q1=1, Q2=2, Q3=3, Q4=4
- NO need for strptime() - Date is already properly typed!

YEAR-OVER-YEAR (YoY) / GROWTH RATE QUERIES:
For queries asking about YoY growth, trends, or comparisons between years:

DYNAMIC YoY Pattern (DO NOT hardcode years!):
WITH sales_by_period AS (
    SELECT
        Category,
        YEAR(Date) as year,
        SUM(Amount) as revenue
    FROM amazon_sales
    WHERE Status != 'Cancelled' AND Amount > 0
        AND QUARTER(Date) = <extract quarter from query>
        AND UPPER("ship-state") LIKE <extract region from query if mentioned>
    GROUP BY Category, year
),
yoy_calc AS (
    SELECT
        curr.Category,
        curr.year as current_year,
        prev.year as previous_year,
        curr.revenue as current_revenue,
        prev.revenue as previous_revenue,
        ((curr.revenue - prev.revenue) / prev.revenue * 100) as yoy_growth_pct
    FROM sales_by_period curr
    LEFT JOIN sales_by_period prev
        ON curr.Category = prev.Category
        AND curr.year = prev.year + 1
)
SELECT Category, current_year, previous_year, yoy_growth_pct
FROM yoy_calc
WHERE previous_revenue IS NOT NULL
ORDER BY yoy_growth_pct DESC
LIMIT 1;

DYNAMIC PARAMETER EXTRACTION:
- "Q3" → WHERE QUARTER(Date) = 3
- "Q1 2022" → WHERE QUARTER(Date) = 1 AND YEAR(Date) = 2022
- "North region" → AND UPPER("ship-state") LIKE '%NORTH%'
- "Maharashtra" → AND UPPER("ship-state") = 'MAHARASHTRA'
- DO NOT hardcode years - use self-joins to find consecutive years automatically

Common Query Patterns:
- Top categories: SELECT Category, SUM(Amount) as total FROM amazon_sales WHERE Status != 'Cancelled' GROUP BY Category ORDER BY total DESC LIMIT 10
- By region: SELECT "ship-state", COUNT(*) as orders FROM amazon_sales WHERE Status != 'Cancelled' GROUP BY "ship-state" ORDER BY orders DESC
- Specific quarter: WHERE QUARTER(strptime(Date, '%m-%d-%y')) = 3
- Specific year: WHERE YEAR(strptime(Date, '%m-%d-%y')) = 2022

For SUMMARIZATION queries: Set query_type to "summarization" and sql_query to null
For Q&A queries: Generate a valid, executable DuckDB SQL query

Return ONLY a JSON object (no markdown, no explanation):
{{
  "query_type": "summarization" or "qa",
  "structured_query": "description of what data is needed",
  "sql_query": "valid DuckDB SQL query or null for summarization",
  "required_tables": ["list", "of", "tables"]
}}

User Query: {state['user_query']}"""

            response = self.llm.invoke([
                SystemMessage(content=system_prompt)
            ])

            # Parse the response
            response_text = response.content

            # Extract JSON from response (handling potential markdown formatting)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            try:
                parsed_response = json.loads(response_text)
            except json.JSONDecodeError:
                # If parsing fails, create a simple fallback
                parsed_response = {
                    "query_type": "qa",
                    "structured_query": state['user_query'],
                    "sql_query": None,
                    "required_tables": tables
                }

            # Update state
            state['query_type'] = parsed_response.get('query_type', 'qa')
            state['structured_query'] = parsed_response.get('structured_query', '')
            state['sql_query'] = parsed_response.get('sql_query', '')
            state['metadata']['query_resolution'] = {
                'required_tables': parsed_response.get('required_tables', []),
                'success': True
            }

            logger.info(f"[{self.name}] Query resolved - Type: {state['query_type']}")

        except Exception as e:
            logger.error(f"[{self.name}] Error: {str(e)}")
            state['errors'].append(f"Query Resolution Error: {str(e)}")
            state['metadata']['query_resolution'] = {'success': False, 'error': str(e)}

        return state


class DataExtractionAgent:
    """
    Agent 2: Executes queries and extracts relevant data
    """

    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.name = "DataExtractionAgent"

    def run(self, state: AgentState) -> AgentState:
        """
        Execute SQL query and extract data
        """
        logger.info(f"[{self.name}] Extracting data...")

        try:
            extracted_data = {}

            if state['query_type'] == 'summarization':
                # For summarization, get comprehensive statistics
                summary_stats = self.data_processor.get_summary_statistics()
                top_categories = self.data_processor.get_top_categories(10)
                regional_performance = self.data_processor.get_regional_performance()

                extracted_data = {
                    'summary_statistics': summary_stats,
                    'top_categories': top_categories.to_dict('records') if not top_categories.empty else [],
                    'regional_performance': regional_performance.to_dict('records')[:10] if not regional_performance.empty else [],
                    'data_type': 'summary'
                }

            elif state['sql_query']:
                # For Q&A, execute the specific SQL query
                try:
                    result_df = self.data_processor.execute_query(state['sql_query'])
                    row_count = len(result_df)

                    # Get dataset metadata for context
                    dataset_metadata = self.data_processor.get_dataset_metadata()

                    extracted_data = {
                        'query_result': result_df.to_dict('records'),
                        'row_count': row_count,
                        'columns': list(result_df.columns),
                        'data_type': 'query_result',
                        'dataset_metadata': dataset_metadata
                    }

                    # If query returned empty results, treat it as a data limitation case
                    if row_count == 0:
                        logger.warning(f"[{self.name}] Query returned 0 rows - likely data limitation")
                        extracted_data['empty_result'] = True
                        extracted_data['user_query'] = state['user_query']

                        # Provide fallback data
                        fallback_df = self.data_processor.get_top_categories(10)
                        extracted_data['fallback_data'] = fallback_df.to_dict('records')
                        extracted_data['fallback_type'] = 'category_stats'
                except Exception as sql_error:
                    # If SQL fails, provide fallback data based on query intent
                    logger.warning(f"[{self.name}] SQL query failed, using fallback: {str(sql_error)}")

                    # Provide relevant fallback data
                    if 'category' in state['user_query'].lower():
                        fallback_df = self.data_processor.get_top_categories(10)
                        extracted_data = {
                            'query_result': fallback_df.to_dict('records'),
                            'row_count': len(fallback_df),
                            'columns': list(fallback_df.columns),
                            'data_type': 'query_result',
                            'fallback': True,
                            'original_error': str(sql_error)
                        }
                    elif 'region' in state['user_query'].lower() or 'state' in state['user_query'].lower():
                        fallback_df = self.data_processor.get_regional_performance()
                        extracted_data = {
                            'query_result': fallback_df.head(10).to_dict('records'),
                            'row_count': 10,
                            'columns': list(fallback_df.columns),
                            'data_type': 'query_result',
                            'fallback': True,
                            'original_error': str(sql_error)
                        }
                    else:
                        # Generic fallback - summary statistics
                        summary_stats = self.data_processor.get_summary_statistics()
                        extracted_data = {
                            'query_result': [summary_stats.get('amazon_sales', {})],
                            'row_count': 1,
                            'columns': list(summary_stats.get('amazon_sales', {}).keys()),
                            'data_type': 'query_result',
                            'fallback': True,
                            'original_error': str(sql_error)
                        }
            else:
                # No SQL query provided (complex query detected), use fallback immediately
                logger.info(f"[{self.name}] No SQL query - using smart fallback for complex query")

                # Provide relevant fallback data based on query intent
                if 'category' in state['user_query'].lower():
                    fallback_df = self.data_processor.get_top_categories(10)
                    extracted_data = {
                        'query_result': fallback_df.to_dict('records'),
                        'row_count': len(fallback_df),
                        'columns': list(fallback_df.columns),
                        'data_type': 'query_result',
                        'fallback': True,
                        'fallback_reason': 'Complex query - showing top categories instead'
                    }
                elif 'region' in state['user_query'].lower() or 'state' in state['user_query'].lower():
                    fallback_df = self.data_processor.get_regional_performance()
                    extracted_data = {
                        'query_result': fallback_df.head(10).to_dict('records'),
                        'row_count': 10,
                        'columns': list(fallback_df.columns),
                        'data_type': 'query_result',
                        'fallback': True,
                        'fallback_reason': 'Complex query - showing regional performance instead'
                    }
                else:
                    # Generic fallback - summary statistics
                    summary_stats = self.data_processor.get_summary_statistics()
                    extracted_data = {
                        'query_result': [summary_stats.get('amazon_sales', {})],
                        'row_count': 1,
                        'columns': list(summary_stats.get('amazon_sales', {}).keys()),
                        'data_type': 'query_result',
                        'fallback': True,
                        'fallback_reason': 'Complex query - showing summary statistics instead'
                    }

            state['extracted_data'] = extracted_data
            state['metadata']['data_extraction'] = {
                'success': True,
                'records_extracted': len(extracted_data.get('query_result', extracted_data.get('top_categories', []))),
                'fallback_used': extracted_data.get('fallback', False)
            }

            logger.info(f"[{self.name}] Data extracted successfully")

        except Exception as e:
            logger.error(f"[{self.name}] Error: {str(e)}")
            state['errors'].append(f"Data Extraction Error: {str(e)}")
            state['metadata']['data_extraction'] = {'success': False, 'error': str(e)}
            # Provide empty data to allow workflow to continue
            state['extracted_data'] = {'data_type': 'error', 'error': str(e)}

        return state


class ValidationAgent:
    """
    Agent 3: Validates extracted data and ensures quality
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.name = "ValidationAgent"

    def run(self, state: AgentState) -> AgentState:
        """
        Validate extracted data and check for inconsistencies
        """
        logger.info(f"[{self.name}] Validating data...")

        try:
            validation_result = {
                'is_valid': True,
                'confidence': 1.0,
                'issues': [],
                'recommendations': []
            }

            extracted_data = state.get('extracted_data', {})

            # Basic validation checks
            if not extracted_data or extracted_data.get('data_type') == 'error':
                validation_result['is_valid'] = False
                validation_result['confidence'] = 0.0
                validation_result['issues'].append("No data extracted or extraction failed")

            # Check data completeness
            if state['query_type'] == 'summarization':
                required_keys = ['summary_statistics', 'top_categories', 'regional_performance']
                missing_keys = [key for key in required_keys if key not in extracted_data]
                if missing_keys:
                    validation_result['issues'].append(f"Missing data components: {missing_keys}")
                    validation_result['confidence'] *= 0.8

            elif state['query_type'] == 'qa':
                if 'query_result' in extracted_data:
                    if len(extracted_data['query_result']) == 0:
                        validation_result['recommendations'].append(
                            "Query returned no results. Consider broadening search criteria."
                        )
                        validation_result['confidence'] *= 0.7

            # Use LLM for semantic validation
            if extracted_data and extracted_data.get('data_type') != 'error':
                validation_prompt = f"""As a data validation expert, review this extracted data:

User Query: {state['user_query']}
Extracted Data: {json.dumps(extracted_data, indent=2, default=str)[:1000]}...

Check if:
1. The data seems relevant to the user's question
2. There are any obvious anomalies or inconsistencies
3. The data quality is sufficient to answer the query

Provide a brief validation assessment (2-3 sentences)."""

                validation_response = self.llm.invoke([
                    SystemMessage(content=validation_prompt)
                ])

                validation_result['llm_assessment'] = validation_response.content

            state['validation_result'] = validation_result
            state['metadata']['validation'] = {
                'success': True,
                'is_valid': validation_result['is_valid'],
                'confidence': validation_result['confidence']
            }

            logger.info(f"[{self.name}] Validation complete - Valid: {validation_result['is_valid']}")

        except Exception as e:
            logger.error(f"[{self.name}] Error: {str(e)}")
            state['errors'].append(f"Validation Error: {str(e)}")
            state['metadata']['validation'] = {'success': False, 'error': str(e)}
            # Allow workflow to continue with default validation
            state['validation_result'] = {
                'is_valid': True,
                'confidence': 0.5,
                'issues': [str(e)]
            }

        return state


class ResponseGenerationAgent:
    """
    Agent 4: Generates natural language response from validated data
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.name = "ResponseGenerationAgent"

    def run(self, state: AgentState) -> AgentState:
        """
        Generate human-readable response
        """
        logger.info(f"[{self.name}] Generating response...")

        try:
            extracted_data = state.get('extracted_data', {})
            validation = state.get('validation_result') or {'is_valid': True, 'confidence': 1.0, 'issues': []}

            # Create context for response generation
            is_fallback = extracted_data.get('fallback', False)
            has_error = extracted_data.get('data_type') == 'error'
            empty_result = extracted_data.get('empty_result', False)
            dataset_metadata = extracted_data.get('dataset_metadata', {})

            system_prompt = """You are a Retail Analytics Expert creating insights for business executives.

Your task is to generate clear, concise, and actionable insights based on the extracted data.

Guidelines:
1. Start with a direct answer to the user's question
2. Include specific numbers and percentages from the data
3. Highlight key trends or patterns
4. Use business-friendly language
5. Keep the response concise (3-5 sentences for Q&A, 1 paragraph for summaries)
6. If query returned empty results, EXPLAIN WHY using dataset metadata
7. If the exact query couldn't be answered, provide the most relevant alternative insights
8. ALWAYS provide what you CAN show from the data

Format your response in a professional, easy-to-read manner."""

            fallback_note = ""
            if empty_result:
                # Analyze WHY the query returned empty
                date_range = dataset_metadata.get('date_range', {})
                user_query_lower = state['user_query'].lower()

                reasons = []
                if 'yoy' in user_query_lower or 'year-over-year' in user_query_lower or 'year over year' in user_query_lower:
                    if date_range.get('unique_years', 0) < 2:
                        reasons.append(f"YoY comparison requires at least 2 years of data, but dataset only contains {date_range.get('available_years', 'limited')} data")

                if 'q3' in user_query_lower:
                    available_quarters = date_range.get('available_quarters', '')
                    if 'Q3' not in available_quarters:
                        reasons.append(f"Q3 data requested but not available. Dataset contains: {available_quarters}")

                if 'q4' in user_query_lower:
                    available_quarters = date_range.get('available_quarters', '')
                    if 'Q4' not in available_quarters:
                        reasons.append(f"Q4 data requested but not available. Dataset contains: {available_quarters}")

                if reasons:
                    fallback_note = f"\nDATA LIMITATION - Query returned empty because:\n" + "\n".join(f"- {r}" for r in reasons)
                    fallback_note += f"\n\nDataset coverage: {date_range.get('min_date', 'Unknown')} to {date_range.get('max_date', 'Unknown')}"
                    fallback_note += f"\nAvailable periods: {date_range.get('available_quarters', 'Unknown')}"
                    fallback_note += f"\n\nInstead, showing relevant insights from available data ({date_range.get('available_quarters', 'available periods')})."
                else:
                    fallback_note = f"\nNOTE: Query returned no results. This might be due to specific filters not matching any data."
            elif is_fallback:
                fallback_note = f"\nNOTE: The original query couldn't be executed as specified. Providing alternative relevant data instead."
            elif has_error:
                fallback_note = f"\nNOTE: Data extraction encountered an error. Provide the best possible response acknowledging the limitation."

            # Prepare data for response
            data_to_show = extracted_data.get('query_result', [])
            if empty_result and extracted_data.get('fallback_data'):
                data_to_show = extracted_data.get('fallback_data', [])

            user_prompt = f"""User Query: {state['user_query']}

Query Type: {state['query_type']}
{fallback_note}

Data to Present:
{json.dumps(data_to_show, indent=2, default=str)[:2000]}

Validation Status:
- Valid: {validation.get('is_valid', True)}
- Confidence: {validation.get('confidence', 1.0)}
- Issues: {validation.get('issues', [])}

Generate a clear, professional response. Start by explaining the data limitation (if any), then provide actionable insights from the available data."""

            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            state['final_response'] = response.content
            state['metadata']['response_generation'] = {'success': True}

            logger.info(f"[{self.name}] Response generated successfully")

        except Exception as e:
            logger.error(f"[{self.name}] Error: {str(e)}")
            state['errors'].append(f"Response Generation Error: {str(e)}")
            state['final_response'] = f"I encountered an error while generating the response: {str(e)}"
            state['metadata']['response_generation'] = {'success': False, 'error': str(e)}

        return state
