# Stage 1: Build dependencies
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target=/app/dependencies

# Stage 2: Create runtime environment
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /app/dependencies /app/dependencies
COPY . .
ENV PYTHONPATH=/app/dependencies
CMD ["python", "app.py"]
