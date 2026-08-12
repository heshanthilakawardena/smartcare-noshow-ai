FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Include /app and /app/server in Python's search path
ENV PYTHONPATH=/app:/app/server:$PYTHONPATH

COPY requirements.txt .
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt
COPY . .

EXPOSE 8080

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120 server.flask_server:app