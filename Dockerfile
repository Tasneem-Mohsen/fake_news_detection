
FROM python:3.10


WORKDIR /app


COPY req.txt .


RUN pip install --upgrade pip && pip install --no-cache-dir -r req.txt


COPY . .


CMD ["python", "app/main.py"]
