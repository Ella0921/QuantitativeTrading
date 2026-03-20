FROM python:3.11-slim

LABEL description="Quantitative Trading ML Pipeline"

WORKDIR /app

# System deps for pandas/numpy/psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/raw data/features data/logs models results

# Default: run the daily pipeline once (for testing / CI smoke test)
CMD ["python", "dags/stock_pipeline.py", "--run-local", "--ticker", "^TWII"]
