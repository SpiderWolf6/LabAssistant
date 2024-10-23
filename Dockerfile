# Use a lightweight Python base image
FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

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
CMD ["chainlit", "run", "rag_llm.py", "--port", "8000"]
