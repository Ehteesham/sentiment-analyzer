FROM python:3.12-slim

LABEL maintainer="Ansari Ehteesham Aqeel" \
      description="A Python based Web application for Sentiment Analysis" \
      version="1.0"


RUN apt update -y 
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m nltk.downloader punkt_tab stopwords
EXPOSE 5000
CMD ["python3", "app.py"]