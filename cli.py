from importlib.metadata import distribution
import boto3
import time
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import glob
import os
import json

with open('config.json') as f:
  config = json.load(f)

bucket = "my-sagemaker-01"
#train_file_name = "distributed-test" + time.strftime("-%Y-%m-%d-%H-%M-%S", time.gmtime()) + ".csv"
train_file_dir = "distributed-test" + time.strftime("-%Y-%m-%d-%H-%M-%S", time.gmtime())
role = config['role']
image_uri = config['image_uri']
#image_uri = "633004b95f8c"
channel = 'train'

s3_cli = boto3.client(
  aws_access_key_id=config['aws_access_key_id'],
  aws_secret_access_key=config['aws_secret_access_key'] ,
  service_name='s3', 
  region_name='us-east-1')

#s3_cli.upload_file('train1.csv', bucket, "{}/{}".format('train_data', train_file_name))

files = glob.glob('train*.csv')
for file in files:
  s3_cli.upload_file(file, bucket, "{}/{}".format(train_file_dir, os.path.basename(file)))


estimator = Estimator(
    image_uri=image_uri,
    role=role,
    #instance_type="local",
    #instance_type="ml.c4.2xlarge",
    instance_type="ml.p3.16xlarge",
    instance_count=3,
    #instance_count=1,
    distribution={ "smdistributed": { "dataparallel": { "enabled": True } } }
)

train_inputs = TrainingInput(
  s3_data="s3://{}/{}".format(bucket, train_file_dir),
  distribution='ShardedByS3Key'
  )
res = estimator.fit({channel: train_inputs})

print(res)