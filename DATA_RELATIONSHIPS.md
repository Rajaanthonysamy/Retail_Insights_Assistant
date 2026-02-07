# 🔗 Data Relationships & Integration Map
## Retail Sales Datasets - ERD & Referential Integrity Analysis

**Date:** February 2026
**Purpose:** Document relationships between all 7 CSV files

---

## 📊 Entity Relationship Diagram (ERD)

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DATASET RELATIONSHIP MAP                           │
└─────────────────────────────────────────────────────────────────────┘

                 ┌──────────────────────────┐
                 │   May-2022 Pricing       │
                 │   PK: Sku (1,330)        │◄─────┐
                 │   SKU Format: Os206_3141 │      │ 100% Match
                 └──────────────────────────┘      │ (Same SKUs)
                         ▲                         │
                         │ ⚠️ NO MATCH             │
                         │ (0% overlap)    ┌───────┴──────────┐
                         │                 │  March-2021      │
     ┌───────────────────┴───────┐         │  PK: Sku (1,330) │
     │                           │         └──────────────────┘
     │                           │
┌────▼────────────────┐  ┌───────▼──────────────┐
│ Amazon Sale Report  │  │ Sale Report (Inv)    │
│ PK: Order ID        │◄─┤ PK: SKU Code (9,170) │
│ FK: SKU (7,195)     │  │ SKU Format: JNE3908  │
│ SKU: JNE3908-KR-XS  │  └──────────────────────┘
└─────────────────────┘          ▲
         │                       │
         │ 51% Match             │ 92% Match
         │ (3,699 SKUs)          │ (6,617 SKUs)
         │                       │
         ▼                       │
┌────────────────────────────────┴──┐
│ International Sale Report         │
│ PK: INDEX (auto)                  │
│ FK: SKU (4,598), Style            │
│ SKU Format: JNE3908-KR-XS         │
└───────────────────────────────────┘

┌──────────────────┐    ┌─────────────────┐
│ Cloud Warehouse  │    │ Expense IIGF    │
│ (50 rows)        │    │ (17 rows)       │
│ ⚠️ STANDALONE    │    │ ⚠️ STANDALONE   │
└──────────────────┘    └─────────────────┘
```

---

## 🔑 Primary Keys & Foreign Keys

### 1. Amazon Sale Report (PRIMARY DATASET)
- **Primary Key:** `Order ID` (⚠️ Only 120,378 unique out of 128,975 - has duplicates!)
- **Foreign Keys:**
  - `SKU` → Links to Inventory (92% match)
  - `SKU` → ⚠️ NO LINK to Pricing (0% match)
- **Unique Values:**
  - 7,195 unique SKUs
  - 9 categories
  - 91 unique dates

### 2. Sale Report (Inventory)
- **Primary Key:** `SKU Code` (9,170 unique)
- **Foreign Keys:** None direct
- **Links:**
  - ✅ Amazon Sales (via SKU) - 6,617 matches
  - ❌ Pricing catalogs - 0 matches

### 3. International Sale Report
- **Primary Key:** INDEX (auto-increment)
- **Foreign Keys:**
  - `SKU` → Links to Amazon Sales (51% match)
  - `Style` → No external reference
- **Unique Values:**
  - 4,598 unique SKUs
  - No unique ID field (data quality issue)

### 4. May-2022 Pricing Catalog
- **Primary Key:** `Sku` (1,330 unique)
- **Foreign Keys:** None
- **Relationships:**
  - ✅ March-2021 Pricing (100% match)
  - ❌ ALL sales data (0% match - different SKU format!)

### 5. March-2021 Pricing Catalog
- **Primary Key:** `Sku` (1,330 unique)
- **Foreign Keys:** None
- **Relationships:**
  - ✅ May-2022 Pricing (100% match - price tracking)

### 6. Cloud Warehouse Comparison
- **Primary Key:** None (comparison table)
- **Standalone:** No foreign keys

### 7. Expense IIGF
- **Primary Key:** None
- **Standalone:** Financial tracking only

---

## ⚠️ Referential Integrity Issues

### CRITICAL Issues:

#### 1. **SKU Format Mismatch** 🚨
**Problem:** Different SKU naming conventions prevent linking

| Dataset | SKU Format | Example |
|---------|-----------|---------|
| Amazon Sales | `PRODUCT-COLOR-SIZE` | `JNE3908-KR-XS` |
| Inventory | `PRODUCT-COLOR-SIZE` | `SET377-KR-NP-XS` |
| International | `PRODUCT-COLOR-SIZE` | `AN204-PURPLE-XL` |
| May-2022 Pricing | `Style_Number_Size` | `Os206_3141_S` |
| March-2021 Pricing | `Style_Number_Size` | `Os206_3141_S` |

**Impact:**
- ❌ Cannot link sales to pricing
- ❌ Cannot calculate profit margins
- ❌ Cannot validate sale prices against catalog

**Solution:**
- Create SKU mapping table
- Standardize SKU format across all systems
- Use Style ID as alternative linking field

#### 2. **Duplicate Order IDs** 🚨
**Problem:** Amazon Sale Report has 128,975 rows but only 120,378 unique Order IDs

**Impact:**
- 8,597 duplicate orders (6.7%)
- Revenue double-counting risk
- Data quality issue

**Solution:**
- Investigate duplicates
- Deduplicate based on business rules
- Add unique constraint

#### 3. **No Unique ID in International Sales** ⚠️
**Problem:** International dataset has no primary key

**Impact:**
- Cannot uniquely identify transactions
- Difficult to track/update records

**Solution:**
- Add auto-increment ID
- Use composite key (DATE + CUSTOMER + SKU)

#### 4. **Orphaned Records**
- **7,195 Amazon SKUs** have no pricing data (100%)
- **9,170 Inventory SKUs** have no pricing data (100%)
- **All International SKUs** have no pricing data (100%)

---

## ✅ Working Relationships

### 1. Amazon ↔ Inventory (92% Match)
**Linking Field:** `SKU` ↔ `SKU Code`

```sql
SELECT
    a.Order_ID,
    a.SKU,
    a.Amount,
    i.Stock,
    i.Category as Inventory_Category
FROM Amazon_Sales a
LEFT JOIN Inventory i ON a.SKU = i.SKU_Code
WHERE i.SKU_Code IS NOT NULL
-- Returns: 6,617 matched SKUs
```

**Use Cases:**
- ✅ Check inventory levels for sold products
- ✅ Track stock movement
- ✅ Identify out-of-stock items

### 2. Amazon ↔ International (51% Match)
**Linking Field:** `SKU` ↔ `SKU`

```sql
SELECT
    'Amazon' as channel,
    SKU,
    SUM(Amount) as revenue
FROM Amazon_Sales
WHERE SKU IN (SELECT SKU FROM International_Sales)
GROUP BY SKU

UNION ALL

SELECT
    'International' as channel,
    SKU,
    SUM(CAST(GROSS_AMT AS FLOAT)) as revenue
FROM International_Sales
WHERE SKU IN (SELECT SKU FROM Amazon_Sales)
GROUP BY SKU
-- Returns: 3,699 common products across channels
```

**Use Cases:**
- ✅ Compare domestic vs international sales
- ✅ Identify best-selling products globally
- ✅ Unified product performance

### 3. May-2022 ↔ March-2021 (100% Match)
**Linking Field:** `Sku` ↔ `Sku`

```sql
SELECT
    m.Sku,
    m.Category,
    m.TP as Current_Price,
    p.TP_1 as Previous_Price,
    ((m.TP - p.TP_1) / p.TP_1 * 100) as Price_Change_Pct
FROM May_2022 m
JOIN March_2021 p ON m.Sku = p.Sku
-- Returns: 1,330 products with price history
```

**Use Cases:**
- ✅ Track price changes over time
- ✅ Analyze pricing strategy
- ✅ Identify markdown/markup patterns

---

## 🔧 Data Integration Recommendations

### Phase 1: Fix SKU Mapping (CRITICAL)

**Option A: Create Mapping Table**
```sql
CREATE TABLE SKU_Mapping (
    Sales_SKU VARCHAR(50),     -- JNE3908-KR-XS
    Pricing_SKU VARCHAR(50),   -- Os206_3141_S
    Style_ID VARCHAR(50),
    Mapping_Type VARCHAR(20),  -- 'Exact', 'Fuzzy', 'Manual'
    Confidence FLOAT,
    PRIMARY KEY (Sales_SKU, Pricing_SKU)
);
```

**Option B: Standardize SKU Format**
- Convert all to: `STYLE_VARIANT_SIZE`
- Example: `JNE3908-KR-XS` → `JNE3908_KR_XS`
- Example: `Os206_3141_S` → `OS206_3141_S`

**Option C: Use Style ID as Link**
- Both datasets might have Style/Style ID
- Map SKU → Style → Pricing

### Phase 2: Create Master Tables

**Master Product Catalog:**
```sql
CREATE VIEW Master_Products AS
SELECT
    COALESCE(i.SKU_Code, m.Sku) as Product_ID,
    i.Category as Inventory_Category,
    m.Category as Pricing_Category,
    i.Stock,
    m.TP as Trade_Price,
    m.Amazon_MRP,
    m.Myntra_MRP,
    m.Flipkart_MRP
FROM Inventory i
FULL OUTER JOIN May_2022_Pricing m
    ON i.SKU_Code = m.Sku;  -- ⚠️ Will be empty until SKU mapping fixed
```

**Unified Sales View:**
```sql
CREATE VIEW Unified_Sales AS
SELECT
    'Amazon' as Channel,
    Order_ID as Transaction_ID,
    SKU,
    Date as Sale_Date,
    Amount as Revenue,
    Status,
    ship_state as Location
FROM Amazon_Sales

UNION ALL

SELECT
    'International' as Channel,
    CAST(INDEX as VARCHAR) as Transaction_ID,
    SKU,
    DATE as Sale_Date,
    CAST(GROSS_AMT as FLOAT) as Revenue,
    NULL as Status,
    NULL as Location
FROM International_Sales;
```

### Phase 3: Data Quality Fixes

1. **Deduplicate Amazon Orders**
   ```sql
   DELETE FROM Amazon_Sales
   WHERE Order_ID IN (
       SELECT Order_ID
       FROM Amazon_Sales
       GROUP BY Order_ID
       HAVING COUNT(*) > 1
   )
   AND Status = 'Cancelled';  -- Keep non-cancelled version
   ```

2. **Add Unique ID to International**
   ```sql
   ALTER TABLE International_Sales
   ADD COLUMN Transaction_ID INT AUTO_INCREMENT PRIMARY KEY;
   ```

3. **Validate Prices**
   ```sql
   -- Flag sales where Amount doesn't match pricing catalog
   SELECT a.*, p.Amazon_MRP
   FROM Amazon_Sales a
   JOIN May_2022_Pricing p ON a.SKU = p.Sku
   WHERE ABS(a.Amount - p.Amazon_MRP) > 100;  -- >₹100 difference
   ```

---

## 📊 Integration Priority Matrix

| Integration | Business Value | Technical Difficulty | Priority | Status |
|-------------|---------------|---------------------|----------|--------|
| Amazon ↔ Inventory | High | Low (92% match) | 🟢 P0 | ✅ Working |
| Amazon ↔ International | Medium | Medium (51% match) | 🟡 P1 | ⚠️ Partial |
| Sales ↔ Pricing | **CRITICAL** | **High (0% match)** | 🔴 **P0** | ❌ **BROKEN** |
| May-22 ↔ Mar-21 | Medium | Low (100% match) | 🟡 P2 | ✅ Working |
| Warehouse Integration | Low | Low | 🟢 P3 | N/A |
| Expense Integration | Low | Low | 🟢 P3 | N/A |

---

## 🎯 Action Items

### Immediate (This Week):
1. ❌ **CRITICAL**: Investigate SKU mismatch between sales and pricing
2. ❌ Obtain SKU mapping table from business/IT
3. ❌ Deduplicate Amazon Order IDs
4. ❌ Add primary key to International Sales

### Short-term (This Month):
5. ⚠️ Create unified product master
6. ⚠️ Build integrated sales view
7. ⚠️ Implement data validation rules
8. ⚠️ Document data lineage

### Long-term (This Quarter):
9. 📋 Standardize SKU format across all systems
10. 📋 Implement referential integrity constraints
11. 📋 Create automated data quality checks
12. 📋 Build comprehensive data warehouse

---

## 📝 Interview Talking Points

**When discussing this analysis:**

✅ **Good News:**
- "We successfully mapped relationships between 5 of 7 datasets"
- "92% of Amazon SKUs match inventory - good data quality"
- "Pricing catalogs are consistent (100% match)"

⚠️ **Challenges Identified:**
- "Discovered critical SKU format mismatch preventing price validation"
- "Found 6.7% duplicate orders requiring investigation"
- "International sales lacks unique identifier"

🔧 **Solutions Proposed:**
- "Recommend creating SKU mapping table as immediate fix"
- "Designed integrated data model with master product catalog"
- "Proposed deduplication and data quality pipeline"

💡 **Business Impact:**
- "Once integrated, can calculate actual profit margins"
- "Enable unified global sales reporting"
- "Improve inventory forecasting accuracy"

---

**Generated by:** Retail Insights Assistant Team
**Last Updated:** February 2026
