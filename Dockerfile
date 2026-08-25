FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY web/ web/
COPY data/ data/

EXPOSE 8765

CMD ["python", "src/serve.py", "--host", "0.0.0.0", "--port", "8765"]
