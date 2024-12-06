# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies if needed (optional)
# RUN apt-get update && apt-get install -y <any-system-deps> && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port (Railway sets the PORT env variable automatically, but we declare a default)
EXPOSE 8000

# Use the PORT environment variable provided by Railway at runtime
ENV PORT=8000

# Command to start the server
# Make sure this matches your actual app structure and filename (app.py) and FastAPI app instance (app)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
