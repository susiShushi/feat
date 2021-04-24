FROM python:3.8-slim

WORKDIR /tmp/work

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

COPY requirements.txt ${PWD}

RUN pip install -r requirements.txt
