FROM python:3.7.15

RUN mkdir -p /opt/ml/

COPY ./train.py /opt/ml/train

ENV PROGRAM_DIR=/opt/ml

ENV PATH=$PATH:$PROGRAM_DIR

RUN pip3 install boto3 watchtower torch

RUN chmod +x $PROGRAM_DIR/train