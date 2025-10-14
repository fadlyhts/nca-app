# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p uploads plots results model

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health

# Run Streamlit app on port 80
ENTRYPOINT ["streamlit", "run", "app_streamlit.py", "--server.port=80", "--server.address=0.0.0.0"]
