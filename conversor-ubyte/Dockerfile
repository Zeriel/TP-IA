FROM python:2.7

RUN mkdir /home/conversor
WORKDIR /home/conversor

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y imagemagick

RUN pip install pillow