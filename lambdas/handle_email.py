import json
import boto3
import os
import email
from email.policy import default as default_policy
import string
import sys
import numpy as np
from hashlib import md5

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1.
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

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
