# Use slim version for smaller image size and reduced attack surface
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files
# PYTHONUNBUFFERED: Ensures logs are flushed immediately (crucial for container logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
# libglib2.0-0: Required for OpenCV (even headless versions often rely on Glib)
# libgomp1: CRITICAL for OnnxRuntime & NumPy threading optimization (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project context
# We copy everything to ensure 'pip install .' has access to the full source tree
COPY . /app/

# Install the package
# 1. Upgrade pip to ensure compatibility with modern wheels
# 2. Install the current package (.) which automatically pulls deps from pyproject.toml
RUN pip install --upgrade pip && \
    pip install .

# Create necessary directories for volume mounting
# This ensures permissions are correct if the user mounts host directories here
RUN mkdir -p data outputs/inference

# Set the entrypoint to your CLI tool
ENTRYPOINT ["bsort"]

# Default command (can be overridden by user)
CMD ["--help"]