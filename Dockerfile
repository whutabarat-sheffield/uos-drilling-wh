# Use an official Python 3 image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code to the container
COPY . .

CMD cd /abyss

# Install Python dependencies
RUN pip install  --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org pip setuptools

# Specify the default command to run the application
#CMD ["python", "abyss/examples-mqtt/listen-continuous.py"]
