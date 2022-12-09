import json
import boto3

def lambda_handler(event, context):
    client = boto3.client('sagemaker')

    instance_name = 'SageMaker-Tutorial'

    #wish to get current status of instance
    status = client.describe_notebook_instance(NotebookInstanceName=instance_name)

    #Start the instance
    try:
        client.stop_notebook_instance(NotebookInstanceName=instance_name)
        print("stopping")
    except:
        try:
            client.start_notebook_instance(NotebookInstanceName=instance_name)
            print("starting")
        except:
            print("did neither")
