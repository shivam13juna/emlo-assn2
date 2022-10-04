import boto3
import botocore


class S3Client:
    def __init__(self, bkt):
        self.bucket = boto3.resource("s3").Bucket(bkt)

    def download_file_from_s3(self, key, out_file):
        try:
            self.bucket.download_file(key, out_file)

        except botocore.exceptions.ClientError as err:
            if err.response["Error"]["Code"] == "404":
                self.log.info("The object does not exist.")
            else:
                raise


if __name__ == "__main__":
    BUCKET_NAME = "test-bucket-emlo-1"
    KEY = "s5/model.script.pt"
    file = "/opt/src/model.script.pt"
    cli = S3Client(BUCKET_NAME)
    cli.download_file_from_s3(KEY, file)

# The above code is a simple example of how to download a file from S3 using boto3. The S3Client class is a wrapper around the boto3 resource. The download_file_from_s3 method is a simple wrapper around the boto3 download_file method. The download_file method takes two arguments, the key of the object to download and the path to the file to write the object to.