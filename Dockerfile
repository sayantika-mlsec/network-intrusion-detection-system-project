# 1. Select the base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first to leverage Docker caching
COPY requirements-serve.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements-serve.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Expose the port FastAPI will run on
EXPOSE 8000

# 7. Define the command to start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]