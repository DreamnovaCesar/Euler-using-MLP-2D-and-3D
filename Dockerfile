FROM python:3.10.0

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python -m", "app.src.main"]
