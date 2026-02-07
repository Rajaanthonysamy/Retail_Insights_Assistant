"""
Check what years and quarters are available in the dataset
"""
from data_processor import DataProcessor

data_processor = DataProcessor()

# Check available years and quarters
check_sql = """
SELECT
    YEAR(Date) as year,
    QUARTER(Date) as quarter,
    COUNT(*) as order_count,
    SUM(Amount) as total_revenue
FROM amazon_sales
WHERE Status != 'Cancelled' AND Amount > 0
GROUP BY YEAR(Date), QUARTER(Date)
ORDER BY year, quarter;
"""

print("=" * 80)
print("Available Data by Year and Quarter")
print("=" * 80)
result = data_processor.execute_query(check_sql)
print(result.to_string(index=False))

# Check if we have Q3 data for multiple years
q3_sql = """
SELECT
    YEAR(Date) as year,
    COUNT(*) as orders,
    SUM(Amount) as revenue
FROM amazon_sales
WHERE Status != 'Cancelled' AND Amount > 0 AND QUARTER(Date) = 3
GROUP BY YEAR(Date)
ORDER BY year;
"""

print("\n" + "=" * 80)
print("Q3 Data by Year")
print("=" * 80)
result_q3 = data_processor.execute_query(q3_sql)
print(result_q3.to_string(index=False))
