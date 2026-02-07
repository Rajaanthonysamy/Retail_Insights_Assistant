"""
Test SQL execution to verify YoY query returns actual results
"""
from data_processor import DataProcessor

# Initialize
data_processor = DataProcessor()

# Test SQL
test_sql = """
WITH sales_by_period AS (
    SELECT
        Category,
        YEAR(Date) as year,
        SUM(Amount) as revenue
    FROM amazon_sales
    WHERE Status != 'Cancelled'
        AND Amount > 0
        AND QUARTER(Date) = 3
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
"""

print("=" * 80)
print("Testing SQL Execution")
print("=" * 80)

try:
    result = data_processor.execute_query(test_sql)
    print("\n✅ SQL executed successfully!")
    print("\n📊 RESULT:")
    print(result.to_string(index=False))
    print("\n" + "=" * 80)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("=" * 80)
