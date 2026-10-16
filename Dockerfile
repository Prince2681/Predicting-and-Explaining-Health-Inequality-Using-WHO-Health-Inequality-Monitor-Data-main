# Use official lightweight Python 3.11 runtime pinned to Debian 12 (bookworm stable)
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Configure apt retries, timeouts, and IPv4 preference to prevent network/DNS stalls
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::ForceIPv4 "true";' >> /etc/apt/apt.conf.d/80-retries

# Install essential runtime packages (OpenMP runtime for XGBoost/LightGBM/CatBoost, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Ensure Linux LF line endings and make entrypoint script executable
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Expose Streamlit dashboard (8501) and MLflow UI (5000)
EXPOSE 8501 5000

# Healthcheck for Streamlit dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default entrypoint runs full pipeline then serves Streamlit web application
ENTRYPOINT ["./entrypoint.sh"]
CMD []
