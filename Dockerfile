# Gunakan image Python yang ringan
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PERBAIKAN DI SINI ---
# Copy SEMUA file dulu (termasuk folder bsort, configs, pyproject.toml)
COPY . .

# Baru jalankan install (sekarang pip bisa menemukan folder bsort)
RUN pip install --no-cache-dir .

# Entrypoint default
ENTRYPOINT ["bsort"]
CMD ["--help"]