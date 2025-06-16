from time import sleep
import os 
from prefect_aws import S3Bucket, AwsCredentials

def create_aws_credentials_block():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id=str(os.environ["ACCESS_ID"]),
        aws_secret_access_key=str(os.environ["ACCESS_PASS"])
    )
    my_aws_creds_obj.save(name="my-aws-credentials", overwrite=True)  # this will be saved in our prefect block


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-credentials") # laoding from prefect server
    my_s3_bucket_obj = S3Bucket(
        bucket_name="mlops-prefect-orchestration",
        credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-prefect-bucket-example", overwrite=True)


if __name__ == "__main__":
    create_aws_credentials_block()
    sleep(5)
    create_s3_bucket_block()

