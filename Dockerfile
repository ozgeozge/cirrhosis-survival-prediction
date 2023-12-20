FROM python:3.10.12-slim 

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY ["predict.py", "model_RF-ADASYN.bin", "/app"]
# Port information will be set via env variables from Heroku
# Also, port information can be passed to the container via -e flag while running the docker container
EXPOSE $PORT 

CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT predict:app
