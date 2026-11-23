FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY licenses ./licenses
RUN python scripts/download_runtime.py
EXPOSE 8000
CMD ["sh", "-c", "python scripts/serve_api.py --host 0.0.0.0 --port ${PORT:-8000} --device cpu"]
