# Use an official Python 3 image as the base image
FROM python:3.10-slim

# Install system dependencies for git and build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Clone the GitHub repository
# Replace GITHUB_REPOSITORY_URL with the actual HTTPS clone URL of your repository
# ARG https://github.com/whutabarat-sheffield/uos-drilling-wh.git
RUN git clone https://github.com/whutabarat-sheffield/uos-drilling-wh.git .

# Change current directory to the abyss directory and install
RUN cd abyss && \
    pip install --no-cache-dir .

# Specify the default command to run the application
CMD ["python", "abyss/examples-mqtt/listen-continuous.py"]
