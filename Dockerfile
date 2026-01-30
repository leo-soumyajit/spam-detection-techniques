# 1. Use standard Python 3.11 (Fixes the version issue)
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy requirements first (for caching speed)
COPY requirements.txt .

# 4. Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the code
COPY . .

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]