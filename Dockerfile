# Dockerfile
# FROM python:3.9.17-bookworm
FROM python:3.10.6-bullseye

#allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

#Copy local code to the container image
ENV APP_HOME /back-end
WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app 