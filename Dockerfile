FROM ubuntu:latest

RUN apt-get update -y && \
    apt-get install -y python3-pip python-dev build-essential

COPY . /firecannon

WORKDIR /firecannon

COPY requirements.txt /firecannon/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:."

RUN pip install -r requirements.txt
RUN python3 setup.py install