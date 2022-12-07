import json
import boto3
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences
import os
import email
from email.policy import default as default_policy

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
vocabulary_length = 9013

def get_prediction(email_body):
    sagemaker = boto3.client('runtime.sagemaker')

    body = [email_body]
    one_hot_email = one_hot_encode(body, vocabulary_length)
    encoded_email = vectorize_sequences(one_hot_email, vocabulary_length)

    response = sagemaker.invoke_endpoint(EndpointName=ENDPOINT_NAME, Body=json.dumps(encoded_email.tolist()))
    prediction = json.loads(response['Body'].read().decode())
    print("Received prediction: ", prediction)

    if int(prediction['predicted_label'][0][0]) == 0:
        return 'HAM', prediction['predicted_probability'][0][0]
    else:
        return 'SPAM', prediction['predicted_probability'][0][0]

def get_s3_file_data(bucket_name, file_key):
    s3 = boto3.client('s3')

    file_data = s3.get_object(Bucket=bucket_name, Key=file_key)['Body'].read().decode()

    return file_data

def send_email(source, destination, subject_line, body):
    ses = boto3.client('ses')
    ses.send_email(Source=source, Destination = {'ToAddresses' : [destination]},
    Message = {
        'Subject' : {'Data' : subject_line},
        'Body' : {
            'Text' : {
                'Data' : body
                }
                }
                })

def lambda_handler(event, context):
    print(json.dumps(event))

    s3_data = event['Records'][0]['s3']
    bucket_name, email_file = s3_data['bucket']['name'], s3_data['object']['key']

    raw_email = get_s3_file_data(bucket_name, email_file)
    email_object = email.message_from_string(raw_email, policy=default_policy)
    sender, receiver = email_object['From'], email_object['To']
    subject, body = email_object['Subject'], email_object.get_body(preferencelist=('plain', 'html')).get_content().strip()
    body_sample = body[:min(len(body), 240)]
    received_date = email_object['Date']

    print("Here is the summary of the email:", sender, receiver, received_date, subject, body_sample)

    spam_status, confidence_score = get_prediction(body)

    response_message = """
    We received your email sent at {received_date} with the subject {subject_line}.

    Here is a 240 character sample of the email body: {email_body_sample}

    The email was catagorized as {spam_status} with {confidence_score}% confidence.

    Best Regards,
    Email Classifier Bot
    """.format(received_date=received_date, subject_line=subject,
                email_body_sample = body_sample, spam_status = spam_status,
                confidence_score = int(confidence_score * 100))

    send_email(receiver, sender, "Re: " + subject, response_message)

    return {}
