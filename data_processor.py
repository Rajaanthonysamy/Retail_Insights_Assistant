"""
Data Processing Layer using DuckDB for efficient querying
"""
import duckdb
import pandas as pd
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data ingestion, preprocessing, and querying using DuckDB"""

    def __init__(self, data_path: str = "Sales Dataset/", db_file: str = "retail_data.duckdb"):
        """
        Initialize DataProcessor with path to sales data

        Args:
            data_path: Path to directory containing sales CSV files
            db_file: Path to persistent DuckDB database file
        """
        self.data_path = data_path
        self.db_file = db_file
        self.conn = duckdb.connect(database=db_file)
        self.tables_loaded = {}
        self._load_datasets()

    def _load_datasets(self):
        """Load all CSV files from data directory into DuckDB (only if tables don't exist)"""
        try:
            # Check if tables already exist in the database
            existing_tables = self.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
            existing_table_names = [table[0] for table in existing_tables]

            # Load Amazon Sale Report
            amazon_file = os.path.join(self.data_path, "Amazon Sale Report.csv")
            if 'amazon_sales' in existing_table_names:
                logger.info("✅ Using existing amazon_sales table from database")
                self.tables_loaded['amazon_sales'] = True
            elif os.path.exists(amazon_file):
                logger.info(f"📥 Loading {amazon_file} into database...")
                self.conn.execute(f"""
                    CREATE TABLE amazon_sales AS
                    SELECT * FROM read_csv_auto('{amazon_file}')
                """)
                self.tables_loaded['amazon_sales'] = True
                logger.info("✅ Amazon sales data loaded successfully")

            # Load Sale Report (Inventory)
            sale_file = os.path.join(self.data_path, "Sale Report.csv")
            if 'inventory' in existing_table_names:
                logger.info("✅ Using existing inventory table from database")
                self.tables_loaded['inventory'] = True
            elif os.path.exists(sale_file):
                logger.info(f"📥 Loading {sale_file} into database...")
                self.conn.execute(f"""
                    CREATE TABLE inventory AS
                    SELECT * FROM read_csv_auto('{sale_file}')
                """)
                self.tables_loaded['inventory'] = True
                logger.info("✅ Inventory data loaded successfully")

            # Load International Sale Report
            intl_file = os.path.join(self.data_path, "International sale Report.csv")
            if 'international_sales' in existing_table_names:
                logger.info("✅ Using existing international_sales table from database")
                self.tables_loaded['international_sales'] = True
            elif os.path.exists(intl_file):
                logger.info(f"📥 Loading {intl_file} into database...")
                self.conn.execute(f"""
                    CREATE TABLE international_sales AS
                    SELECT * FROM read_csv_auto('{intl_file}')
                """)
                self.tables_loaded['international_sales'] = True
                logger.info("✅ International sales data loaded successfully")

            logger.info(f"✅ Database ready with {len(self.tables_loaded)} tables")

        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame

        Args:
            query: SQL query string

        Returns:
            Query results as pandas DataFrame
        """
        try:
            result = self.conn.execute(query).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table"""
        try:
            schema_query = f"DESCRIBE {table_name}"
            schema_df = self.execute_query(schema_query)
            return schema_df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to get schema for {table_name}: {str(e)}")
            return {}

    def get_table_context(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive context about a table for LLM
        Includes: schema, sample data, statistics, and value examples
        """
        try:
            context = {
                'table_name': table_name,
                'schema': [],
                'sample_rows': [],
                'statistics': {},
                'value_examples': {}
            }

            # Get schema
            schema_df = self.execute_query(f"DESCRIBE {table_name}")
            context['schema'] = schema_df.to_dict('records')

            # Get row count
            count_df = self.execute_query(f"SELECT COUNT(*) as total FROM {table_name}")
            context['statistics']['total_rows'] = int(count_df['total'].values[0])

            # Get sample rows (first 3)
            sample_df = self.execute_query(f"SELECT * FROM {table_name} LIMIT 3")
            context['sample_rows'] = sample_df.to_dict('records')

            # Get value examples for key columns
            key_columns = self._get_key_columns(table_name)
            for col in key_columns:
                try:
                    # Get distinct values (limited to 5)
                    distinct_query = f'SELECT DISTINCT "{col}" FROM {table_name} WHERE "{col}" IS NOT NULL LIMIT 5'
                    distinct_df = self.execute_query(distinct_query)
                    context['value_examples'][col] = distinct_df[col].tolist()
                except:
                    pass

            return context

        except Exception as e:
            logger.error(f"Failed to get table context for {table_name}: {str(e)}")
            return {}

    def _get_key_columns(self, table_name: str) -> List[str]:
        """Identify key columns for a table"""
        key_column_mapping = {
            'amazon_sales': ['Category', 'Status', 'ship-state', 'Date'],
            'inventory': ['Category', 'Size', 'Color'],
            'international_sales': ['CUSTOMER', 'Months']
        }
        return key_column_mapping.get(table_name, [])

    def get_available_tables(self) -> List[str]:
        """Get list of available tables"""
        return list(self.tables_loaded.keys())

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics across all datasets"""
        summary = {}

        try:
            # Amazon Sales Summary
            if 'amazon_sales' in self.tables_loaded:
                amazon_stats = self.conn.execute("""
                    SELECT
                        COUNT(*) as total_orders,
                        COUNT(DISTINCT "Order ID") as unique_orders,
                        SUM(Amount) as total_revenue,
                        AVG(Amount) as avg_order_value,
                        COUNT(DISTINCT Category) as total_categories,
                        COUNT(DISTINCT "ship-state") as total_states
                    FROM amazon_sales
                    WHERE Status NOT IN ('Cancelled')
                """).fetchdf()
                summary['amazon_sales'] = amazon_stats.to_dict('records')[0]

            # International Sales Summary
            if 'international_sales' in self.tables_loaded:
                intl_stats = self.conn.execute("""
                    SELECT
                        COUNT(*) as total_transactions,
                        SUM(PCS) as total_pieces,
                        SUM("GROSS AMT") as total_revenue,
                        AVG("GROSS AMT") as avg_transaction_value,
                        COUNT(DISTINCT CUSTOMER) as unique_customers
                    FROM international_sales
                """).fetchdf()
                summary['international_sales'] = intl_stats.to_dict('records')[0]

            # Inventory Summary
            if 'inventory' in self.tables_loaded:
                inventory_stats = self.conn.execute("""
                    SELECT
                        COUNT(*) as total_skus,
                        SUM(Stock) as total_stock,
                        COUNT(DISTINCT Category) as total_categories,
                        COUNT(DISTINCT Color) as total_colors
                    FROM inventory
                """).fetchdf()
                summary['inventory'] = inventory_stats.to_dict('records')[0]

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary statistics: {str(e)}")
            return {}

    def get_top_categories(self, limit: int = 10) -> pd.DataFrame:
        """Get top performing categories by revenue"""
        try:
            query = f"""
                SELECT
                    Category,
                    COUNT(*) as order_count,
                    SUM(Amount) as total_revenue,
                    AVG(Amount) as avg_order_value
                FROM amazon_sales
                WHERE Status NOT IN ('Cancelled')
                GROUP BY Category
                ORDER BY total_revenue DESC
                LIMIT {limit}
            """
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to get top categories: {str(e)}")
            return pd.DataFrame()

    def get_regional_performance(self) -> pd.DataFrame:
        """Get sales performance by region/state"""
        try:
            query = """
                SELECT
                    "ship-state" as state,
                    COUNT(*) as order_count,
                    SUM(Amount) as total_revenue,
                    AVG(Amount) as avg_order_value
                FROM amazon_sales
                WHERE Status NOT IN ('Cancelled')
                    AND "ship-state" IS NOT NULL
                GROUP BY "ship-state"
                ORDER BY total_revenue DESC
            """
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to get regional performance: {str(e)}")
            return pd.DataFrame()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
