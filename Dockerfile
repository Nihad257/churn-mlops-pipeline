# Use a slim Python image (lightweight, fast on your i3)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]