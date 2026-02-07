"""
Test with Q2 data to see month-over-month growth
"""
from data_processor import DataProcessor

data_processor = DataProcessor()

# Since we only have 2022 data, test month-over-month instead
test_sql = """
SELECT
    Category,
    MONTH(Date) as month,
    SUM(Amount) as monthly_revenue
FROM amazon_sales
WHERE Status != 'Cancelled'
    AND Amount > 0
    AND QUARTER(Date) = 2
GROUP BY Category, MONTH(Date)
ORDER BY Category, month;
"""

print("=" * 80)
print("Q2 2022 Monthly Revenue by Category (Actual Available Data)")
print("=" * 80)
result = data_processor.execute_query(test_sql)
print(result.to_string(index=False))
print("\n✅ SQL executes successfully with DATE functions!")
print("=" * 80)
