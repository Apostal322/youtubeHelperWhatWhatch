FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# ffmpeg/ffprobe for the authorized-media path; prlimit (util-linux) bounds their resources.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg util-linux \
    && rm -rf /var/lib/apt/lists/*

# The repository layout is preserved (backend/, data/, benchmarks/) because fixtures and the
# benchmark dataset are resolved relative to the repository root.
COPY backend/pyproject.toml backend/README.md /app/backend/
COPY backend/avd /app/backend/avd
RUN pip install "/app/backend[anthropic]"
COPY backend/alembic.ini /app/backend/alembic.ini
COPY data /app/data
COPY benchmarks /app/benchmarks

RUN useradd --create-home --uid 10001 avd && mkdir -p /app/backend/.media_cache && chown -R avd /app/backend/.media_cache
USER avd
WORKDIR /app/backend
ENV AVD_FIXTURES_DIR=/app/data/fixtures AVD_MEDIA_CACHE_DIR=/app/backend/.media_cache
EXPOSE 8000
CMD ["uvicorn", "avd.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
