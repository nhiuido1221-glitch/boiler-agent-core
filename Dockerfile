FROM python:3.10-slim

# Khong buffer stdout/stderr - xem log realtime tren Render/Railway
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cai dependency truoc de tan dung Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy toan bo source code
COPY . .

# Khong chay bang root de tang bao mat container
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck noi bo container (Render/Railway co the dung rieng health check cua ho)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT','8000') + '/health').read()" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
