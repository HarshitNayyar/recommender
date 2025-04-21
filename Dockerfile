FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    default-libmysqlclient-dev \
    libssl-dev \
    pkg-config \
    netcat-openbsd \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy only requirements first (for better cache use)
COPY requirements.txt .

# Install Python dependencies (torch CPU-only)
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

COPY entrypoint.sh /entrypoint.sh

# Set permissions for entrypoint
RUN chmod +x /entrypoint.sh

# Expose Django port
EXPOSE 8000

# Entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Start the Django dev server
CMD ["python", "backend/manage.py", "runserver", "0.0.0.0:8000"]
