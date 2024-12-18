# Use an official Python 3 image as the base image
ARG PYTHON_VERSION=3.10.15
FROM python:${PYTHON_VERSION}-slim

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Install system dependencies for git and build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Change current directory to the abyss directory and install
RUN cd abyss && \
    pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org .
    # pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118/ .

# Expose the port that the application listens on.
EXPOSE 1883

# Switch to the non-privileged user to run the application.
# USER appuser

# Specify the default command to run the application
CMD ["python", "abyss/examples-mqtt/listen-continuous.py"]
# CMD ["python", "abyss/src/run/uos_depth_estimation_listen-continuous.py"]
# Run the application.
# CMD python listen-continuous.py 