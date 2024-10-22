# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install the required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the working directory
COPY . .

# Expose the port your app will be accessible on (adjust if needed)
EXPOSE 8000

# Command to run the app (assuming rag_llm.py starts your app)
CMD ["python", "rag_llm.py"]
