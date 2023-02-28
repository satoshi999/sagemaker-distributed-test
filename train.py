#!/usr/bin/env python
import os
import boto3
import socket
import time
import watchtower, logging
import glob
import smdistributed.dataparallel.torch.torch_smddp
import torch
import json

with open('config.json') as f:
  config = json.load(f)

def write(data, file):
  with open(file, 'w') as w:
    for d in data:
      w.write(d.join(",") + "\n")

log_cli = boto3.client(
  aws_access_key_id=config['aws_access_key_id'],
  aws_secret_access_key=config['aws_secret_access_key'] ,
  service_name='logs', 
  region_name='us-east-1')

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('sample_logger')
_logger.addHandler(watchtower.CloudWatchLogHandler(boto3_client=log_cli))

ip = socket.gethostbyname(socket.gethostname())

prefix = "/opt/ml/"

train_data_dir = os.path.join(prefix, 'input/data/train/')
model_file = os.path.join(prefix, 'model.csv')
#param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

#with open(param_path) as f:
#  hp = json.load(f)

for key, val in os.environ.items():
  _logger.info('{}: {}'.format(key, val))

files = glob.glob(os.path.join(train_data_dir, '*'))
_logger.info(files)

torch.distributed.init_process_group(backend='smddp')

data = []
with open(files[0]) as f:
  lines = f.readlines()
  length = len(lines)

  _logger.info('{}-dataLength: {}, ip:{}'.format(files[0], str(length), ip))

  for line in lines:
    row = line.rstrip("\n").split(",")

    sum = int(row[0]) + int(row[1])
    now = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

    data.append([row[0], row[1], sum, ip.replace('.', '')])

if torch.distributed.get_rank() != 0:
  torch.distributed.send(torch.tensor(data), 0)

if torch.distributed.get_rank() == 0:
  write(data, model_file)
  r_data = torch.tensor.new_zeros()
  torch.distributed.recv(r_data)
  write(r_data.tolist(), model_file)
  torch.distributed.barrier()