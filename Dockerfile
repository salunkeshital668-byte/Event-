FROM python:3.11-slim

# Install system dependencies required by OpenCV and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 5000

# Use gunicorn as the production server
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
