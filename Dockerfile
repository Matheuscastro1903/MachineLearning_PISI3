FROM python:3.11-slim

WORKDIR /app

COPY . /app/

WORKDIR /app/dash

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]