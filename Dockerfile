FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies for opencv and rembg
RUN apt-get update && \
    apt-get install -y gcc libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements (create requirements.txt if you don't have one)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]