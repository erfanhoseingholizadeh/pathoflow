# 1. Use a lightweight Python base image (Debian-based)
FROM python:3.11-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install System Dependencies (GL1 is the fix from before)
RUN apt-get update && apt-get install -y \
    openslide-tools \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set workdir
WORKDIR /app

# 5. OPTIMIZATION: Install heavy Python libraries FIRST.
# This creates a cached "layer". If you change your code later, 
# Docker will skip this slow step and only run the fast steps below.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch \
    torchvision \
    openslide-python \
    numpy \
    opencv-python-headless \
    pydantic \
    typer \
    matplotlib \
    pytest

# 6. NOW Copy the project configuration and code
COPY pyproject.toml /app/
COPY src /app/src

# 7. Install your actual package (This is now very fast)
RUN pip install --no-cache-dir -e .

# 8. Define the entrypoint
ENTRYPOINT ["pathoflow"]