"""
Configuration module for Retail Insights Assistant
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Data Configuration
DATA_PATH = os.getenv("DATA_PATH", "Sales Dataset/")

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Agent Configuration
MAX_ITERATIONS = 10
AGENT_TIMEOUT = 120  # seconds

# Database Configuration
DB_PATH = ":memory:"  # Use in-memory DuckDB for faster queries

# Logging Configuration
LOG_LEVEL = "INFO"
