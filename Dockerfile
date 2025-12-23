# 1. Base Image
FROM python:3.11-slim

# 2. Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_IO_ENABLE_JASPER=true 

# 3. System Dependencies
RUN apt-get update && apt-get install -y \
    openslide-tools \
    libgl1 \
    libglib2.0-0 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 4. Work Directory
WORKDIR /app

# 5. Copy Project Files (Code + Config)
# We copy BOTH now so pip can see the 'src' folder during install
COPY pyproject.toml /app/
COPY src /app/src

# 6. Install Python Dependencies & Project
# We install in editable mode (-e) so it links correctly
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# 7. Create Mount Points
RUN mkdir -p /models /data /app/outputs

# 8. Entrypoint
ENTRYPOINT ["pathoflow"]