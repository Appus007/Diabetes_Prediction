# Use Python base image
FROM python:3.9-slim

# Install system dependencies required by TenSEAL
RUN apt-get update && \
    apt-get install -y cmake g++ python3-dev git && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
