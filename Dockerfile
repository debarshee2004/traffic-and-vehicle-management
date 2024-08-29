# NOTE: This is an example Dockerfile and not for production.
# Please use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version for smaller image size, change if GPU is needed)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow
RUN pip install tensorflow

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME ML_Project

# Run app.py when the container launches
CMD ["python", "app.py"]