import boto3
import csv

# Set your AWS access key ID and secret access key
ACCESS_KEY = 'AKIA5EX7ZGYJKFOC4EEW'
SECRET_KEY = 'JQ0mgATopRA+Y8kVLSwnwgG2glEm3olAH9C5tMkt'

# Set your S3 bucket name and file name
BUCKET_NAME = 'objectsdatacsv'

# Set your CSV file name and data
CSV_NAME = 'Euler_Data.csv'
FOLDER_NAME = 'Test_1/'
FOLDER_IMAGES = 'Images/'
# Create an S3 client object using your IAM credentials
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

# Create the folder inside your S3 bucket
s3.put_object(Bucket=BUCKET_NAME, Key=FOLDER_NAME)

# Create the folder inside your S3 bucket
s3.put_object(Bucket=BUCKET_NAME, Key=FOLDER_NAME + FOLDER_IMAGES)

# Upload the CSV file to the folder inside your S3 bucket
s3.put_object(Bucket=BUCKET_NAME, Key=FOLDER_NAME + CSV_NAME)

'''# Upload the file to S3
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
s3.download_file(BUCKET_NAME, FILE_NAME, FILE_NAME)'''
