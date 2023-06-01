FROM python:3.11-slim

ENV PYTHONUNBUFFERED True

RUN mkdir /gpt-pdf

COPY . ./gpt-pdf
WORKDIR /gpt-pdf

# RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y python3-dev default-libmysqlclient-dev gcc
RUN pip install -r requirements.txt

# CMD flask run --host=0.0.0.0

#CMD gunicorn --workers $WORKERS \
#  --threads $THREADS \
#  --bind 0.0.0.0:$PORT_APP \
#  --log-level DEBUG \
#  app:app

#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app