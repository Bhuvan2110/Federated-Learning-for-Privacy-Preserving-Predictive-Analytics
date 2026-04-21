# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from build stage
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create non-root user
RUN addgroup --system tm && adduser --system --ingroup tm tm
USER tm

EXPOSE 8080

# Use gunicorn for production; fall back to Flask dev server for local dev
CMD ["python", "app.py"]
# Production override (uncomment and add gunicorn to requirements for prod):
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "app:app"]
