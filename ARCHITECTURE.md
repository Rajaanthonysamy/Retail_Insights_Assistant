# 🏗️ Scalability Architecture for 100GB+ Datasets

## Executive Summary

This document outlines the scalable architecture design for the Retail Insights Assistant when handling datasets exceeding 100GB. The architecture addresses data engineering, storage, retrieval efficiency, model orchestration, and monitoring requirements.

---

## 📊 Current vs. Scalable Architecture

### Current Implementation (< 10GB)
- **Data Storage**: In-memory DuckDB
- **Processing**: Single-machine pandas/DuckDB
- **LLM**: Direct OpenAI API calls
- **UI**: Single Streamlit instance

### Scalable Architecture (100GB+)
- **Data Storage**: Cloud data warehouse + data lake
- **Processing**: Distributed computing (PySpark/Dask)
- **LLM**: Cached, load-balanced API calls
- **UI**: Containerized, horizontally scalable

---

## 🔧 A. Data Engineering & Preprocessing

### 1. Data Ingestion Pipeline

#### Batch Processing
```
Raw CSV Files → Cloud Storage → ETL Pipeline → Data Warehouse
     ↓              ↓                ↓              ↓
  S3/GCS      Staging Layer    PySpark/Dask    BigQuery/Snowflake
```

**Technologies:**
- **Apache Airflow**: Orchestrate ETL workflows
- **PySpark**: Distributed data processing
- **AWS Glue / Azure Data Factory**: Managed ETL service
- **Dask**: Python-native distributed computing

#### Streaming Processing (Optional)
For real-time data:
```
Kafka/Kinesis → Spark Streaming → Real-time Analytics Layer
```

### 2. Data Cleaning & Preprocessing

**PySpark Implementation Example:**
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("RetailDataETL") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Read massive CSV
df = spark.read.csv("s3://retail-data/sales/*.csv", header=True, inferSchema=True)

# Clean and transform
df_cleaned = df \
    .dropna(subset=['Order ID', 'Amount']) \
    .withColumn('Order_Date', to_date('Date', 'MM-dd-yy')) \
    .withColumn('Year', year('Order_Date')) \
    .withColumn('Quarter', quarter('Order_Date'))

# Write to optimized format
df_cleaned.write \
    .partitionBy('Year', 'Quarter') \
    .mode('overwrite') \
    .parquet("s3://retail-data/processed/")
```

### 3. Data Partitioning Strategy

**Time-based Partitioning:**
```
/retail_data/
  /year=2022/
    /quarter=Q1/
    /quarter=Q2/
  /year=2023/
    /quarter=Q1/
```

**Benefits:**
- Query only relevant partitions
- Parallel processing across partitions
- Easier data lifecycle management

---

## 💾 B. Storage & Indexing

### 1. Multi-Tier Storage Architecture

```
┌─────────────────────────────────────────────────┐
│          Query Layer (Fast Access)              │
│  BigQuery / Snowflake / Databricks SQL          │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│      Analytical Layer (Optimized Storage)       │
│  Delta Lake / Parquet Files / Iceberg           │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│        Data Lake (Raw Storage)                  │
│  S3 / Azure Data Lake / Google Cloud Storage    │
└─────────────────────────────────────────────────┘
```

### 2. Recommended Technologies

#### Cloud Data Warehouse (Primary)
**Google BigQuery:**
- Serverless, auto-scaling
- Columnar storage
- Built-in ML capabilities
- Pay-per-query pricing

**Snowflake:**
- Separation of compute and storage
- Automatic optimization
- Data sharing capabilities

#### Data Lake (Raw Storage)
**AWS S3 / GCS:**
- Cost-effective for raw data
- Unlimited scalability
- Integration with processing frameworks

#### Analytical Layer
**Delta Lake:**
```python
# Write data with Delta Lake
df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .partitionBy("year", "quarter") \
    .save("/delta/retail_sales")

# Query with time travel
df_historical = spark.read \
    .format("delta") \
    .option("versionAsOf", "2023-01-01") \
    .load("/delta/retail_sales")
```

**Benefits:**
- ACID transactions
- Time travel (versioning)
- Schema evolution
- Efficient updates/deletes

### 3. Indexing Strategy

**Primary Indices:**
- Order ID, Transaction ID (unique identifiers)
- Date fields (for time-based queries)
- Category, Region (for filtering)

**Secondary Indices:**
- Customer ID
- Product SKU
- State/City

**Materialized Views:**
```sql
CREATE MATERIALIZED VIEW mv_sales_summary AS
SELECT
    DATE_TRUNC('month', order_date) as month,
    category,
    region,
    SUM(amount) as total_revenue,
    COUNT(*) as order_count
FROM sales_data
GROUP BY 1, 2, 3;
```

---

## 🔍 C. Retrieval & Query Efficiency

### 1. Query Optimization Strategies

#### A. Metadata-Based Filtering
```python
class SmartQueryEngine:
    def __init__(self):
        self.metadata_index = {
            'date_ranges': {},
            'category_stats': {},
            'region_stats': {}
        }

    def filter_partitions(self, query_params):
        """Only scan relevant partitions"""
        partitions = []

        if 'date_range' in query_params:
            partitions = self.get_partitions_by_date(
                query_params['date_range']
            )

        return partitions
```

#### B. Semantic Search with Vector Embeddings

**Architecture:**
```
User Query → Embedding Model → Vector Search → Relevant Data Chunks
                                    ↓
                              FAISS/Pinecone
```

**Implementation:**
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Create embeddings for data summaries
embeddings = OpenAIEmbeddings()

# Pre-computed summaries of data partitions
partition_summaries = [
    "Q1 2022 sales data: Electronics category, Western region",
    "Q2 2022 sales data: Clothing category, Eastern region",
    # ... more summaries
]

# Create vector store
vector_store = FAISS.from_texts(partition_summaries, embeddings)

# Query
relevant_partitions = vector_store.similarity_search(
    "Show me electronics sales in Q1",
    k=3
)
```

### 2. RAG (Retrieval-Augmented Generation) Pattern

**Enhanced Architecture:**
```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ Query Resolution    │
│ Agent               │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐      ┌──────────────────┐
│ Semantic Search     │◄─────│  Vector Store    │
│ (Embeddings)        │      │  (FAISS/Pinecone)│
└──────┬──────────────┘      └──────────────────┘
       │
       ▼
┌─────────────────────┐      ┌──────────────────┐
│ Data Extraction     │◄─────│  Data Warehouse  │
│ Agent (SQL Query)   │      │  (BigQuery)      │
└──────┬──────────────┘      └──────────────────┘
       │
       ▼
┌─────────────────────┐
│ LLM Response        │
│ Generation          │
└─────────────────────┘
```

**Code Example:**
```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Query with context
result = qa_chain({
    "query": "What were the top selling categories in Q4 2022?"
})
```

### 3. Query Result Caching

**Redis-based Caching:**
```python
import redis
import hashlib
import json

class QueryCache:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.ttl = 3600  # 1 hour

    def get_cached_result(self, query: str):
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cached = self.redis_client.get(query_hash)

        if cached:
            return json.loads(cached)
        return None

    def cache_result(self, query: str, result: dict):
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.redis_client.setex(
            query_hash,
            self.ttl,
            json.dumps(result)
        )
```

---

## 🤖 D. Model Orchestration

### 1. LLM Query Optimization

#### Prompt Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_system_prompt(query_type: str) -> str:
    """Cache system prompts"""
    return SYSTEM_PROMPTS.get(query_type)
```

#### Batch Processing
```python
async def process_queries_batch(queries: List[str]):
    """Process multiple queries in parallel"""
    tasks = [orchestrator.process_query(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

### 2. Cost Optimization Strategies

**Token Management:**
```python
import tiktoken

class TokenOptimizer:
    def __init__(self, model="gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = 8000  # Reserve tokens for response

    def truncate_context(self, context: str) -> str:
        """Truncate context to fit within token limits"""
        tokens = self.encoding.encode(context)

        if len(tokens) > self.max_tokens:
            # Keep most recent context
            truncated_tokens = tokens[-self.max_tokens:]
            return self.encoding.decode(truncated_tokens)

        return context
```

**Model Selection Strategy:**
```python
def select_model(query_complexity: str) -> str:
    """Choose appropriate model based on complexity"""
    if query_complexity == "simple":
        return "gpt-3.5-turbo"  # Faster, cheaper
    elif query_complexity == "complex":
        return "gpt-4-turbo"     # More capable
    return "gpt-4"
```

### 3. LangChain Integration at Scale

**Optimized Chain:**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache

# Enable caching
set_llm_cache(RedisCache(redis_url="redis://localhost:6379"))

# Create optimized chain
template = PromptTemplate(
    input_variables=["context", "query"],
    template="""
    Context: {context}
    Question: {query}
    Answer concisely:
    """
)

chain = LLMChain(
    llm=llm,
    prompt=template,
    verbose=False
)
```

---

## 📊 E. Monitoring & Evaluation

### 1. Key Metrics

#### Performance Metrics
```python
import time
from prometheus_client import Counter, Histogram

# Define metrics
query_duration = Histogram(
    'query_duration_seconds',
    'Time spent processing query',
    ['query_type']
)

query_counter = Counter(
    'queries_total',
    'Total number of queries',
    ['status']
)

# Track metrics
@query_duration.labels(query_type='qa').time()
def process_query(query: str):
    # ... processing logic
    pass
```

**Tracked Metrics:**
- Query latency (p50, p95, p99)
- Throughput (queries per second)
- Error rate
- Cache hit rate
- LLM API costs
- Data scanned per query

#### Quality Metrics
```python
class QualityMonitor:
    def __init__(self):
        self.metrics = {
            'accuracy_scores': [],
            'confidence_scores': [],
            'user_feedback': []
        }

    def log_response_quality(self, result: dict):
        """Track response quality"""
        self.metrics['confidence_scores'].append(
            result['validation']['confidence']
        )

    def calculate_avg_confidence(self) -> float:
        return sum(self.metrics['confidence_scores']) / \
               len(self.metrics['confidence_scores'])
```

### 2. Error Handling & Fallback Strategies

**Circuit Breaker Pattern:**
```python
from pybreaker import CircuitBreaker

llm_breaker = CircuitBreaker(fail_max=5, timeout_duration=60)

@llm_breaker
def call_llm_api(prompt: str):
    """Call LLM with circuit breaker"""
    try:
        return llm.invoke(prompt)
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        raise
```

**Fallback Strategy:**
```python
def generate_response_with_fallback(query: str):
    try:
        # Try primary LLM
        return call_gpt4(query)
    except Exception:
        # Fallback to GPT-3.5
        try:
            return call_gpt35(query)
        except Exception:
            # Fallback to rule-based response
            return generate_rule_based_response(query)
```

### 3. Logging & Observability

**Structured Logging:**
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log_query(self, query: str, metadata: dict):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'metadata': metadata,
            'type': 'query_log'
        }
        self.logger.info(json.dumps(log_entry))
```

**Monitoring Dashboard:**
- **Grafana**: Visualize metrics from Prometheus
- **DataDog**: Application performance monitoring
- **CloudWatch/Stackdriver**: Cloud-native monitoring

---

## 🎯 Complete Scalable Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│  Streamlit (Load Balanced) + Nginx/API Gateway                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   Multi-Agent Orchestration                      │
│            LangGraph Agents (Containerized, K8s)                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │Query Resolver│ │Data Extractor│ │  Validator   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼──────┐  ┌─────────▼────────┐
│  LLM Services  │  │ Vector Store  │  │  Data Warehouse  │
│   (OpenAI)     │  │(FAISS/Pinecone)│  │  (BigQuery)      │
│  + Cache Layer │  │  + Metadata   │  │  + Delta Lake    │
└────────────────┘  └────────────────┘  └─────────┬────────┘
                                                   │
                                        ┌──────────▼─────────┐
                                        │   Data Lake (S3)   │
                                        │   Raw CSV Files    │
                                        └────────────────────┘

        ┌────────────────────────────────────────┐
        │    Supporting Infrastructure           │
        ├────────────────────────────────────────┤
        │ • Redis Cache (Query Results)          │
        │ • Airflow (ETL Orchestration)          │
        │ • Kafka (Streaming, Optional)          │
        │ • Prometheus + Grafana (Monitoring)    │
        │ • ELK Stack (Logging)                  │
        └────────────────────────────────────────┘
```

---

## 💰 Cost Optimization

### 1. Data Storage Costs
- **Lifecycle Policies**: Move old data to cheaper storage (S3 Glacier)
- **Compression**: Use Parquet/ORC with compression (saves 80-90%)
- **Partitioning**: Only scan relevant partitions

### 2. LLM API Costs
- **Caching**: Reduce redundant API calls by 60-80%
- **Model Selection**: Use GPT-3.5 for simple queries
- **Prompt Optimization**: Reduce token usage
- **Batch Processing**: Lower per-token costs

### 3. Compute Costs
- **Auto-scaling**: Scale down during low traffic
- **Spot Instances**: Use for batch processing (70% savings)
- **Serverless**: Pay only for actual compute time

**Estimated Monthly Costs (100GB dataset, 10K queries/month):**
- Data Storage (S3 + BigQuery): $150-300
- LLM API Costs: $500-1,500
- Compute (K8s cluster): $200-500
- **Total: ~$1,000-2,500/month**

---

## 🚀 Deployment Strategy

### Containerization (Docker)
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retail-insights-assistant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: retail-insights
  template:
    metadata:
      labels:
        app: retail-insights
    spec:
      containers:
      - name: app
        image: retail-insights:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
```

---

## 📝 Summary

This architecture provides:
- ✅ **Horizontal Scalability**: Handle 100GB+ datasets
- ✅ **Cost Efficiency**: Optimized storage and compute
- ✅ **High Performance**: Sub-second query responses
- ✅ **Reliability**: Error handling and fallbacks
- ✅ **Observability**: Comprehensive monitoring
- ✅ **Flexibility**: Cloud-agnostic design

The system can scale from gigabytes to petabytes by leveraging cloud-native services, distributed computing, and intelligent caching strategies.
