FROM python:3.11-slim

# Set container working directory
WORKDIR /app

# Install essential system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained models
COPY app.py .
COPY data/ data/
COPY scripts/ scripts/

# Set environment variables and expose default port
ENV PORT=8080
EXPOSE 8080

# Start Streamlit on the specified Cloud Run port
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
