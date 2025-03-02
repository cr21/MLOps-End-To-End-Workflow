import boto3
import os
import glob

def upload_file_to_s3(file_path, bucket_name, s3_file_name=None):
    """
    Uploads a file to an S3 bucket.

    :param file_path: The local path to the file to be uploaded.
    :param bucket_name: The name of the S3 bucket.
    :param s3_file_name: The name to use for the file in S3. If None, uses the file name from file_path.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Use the original file name if s3_file_name is not provided
    if s3_file_name is None:
        dir_name = os.path.basename(os.path.dirname(file_path))
        s3_file_name = os.path.join(dir_name,os.path.basename(file_path))
        

    # Upload the file
    try:
        s3_client.upload_file(file_path, bucket_name, s3_file_name)
        print(f"File {file_path} uploaded to {bucket_name}/{s3_file_name}.")
    except Exception as e:
        print(f"Failed to upload {file_path} to S3: {e}")

import glob

def remove_files(directory, pattern="*.ckpt"):
    """
    Removes all .ckpt files from the specified directory.

    :param directory: The directory from which to remove .ckpt files.
    """
    print(f"removing checkpoints from {directory}")
    # Construct the pattern for .ckpt files
    pattern = os.path.join(directory, pattern)
    
    # Use glob to find all .ckpt files
    ckpt_files = glob.glob(pattern)
    
    # Remove each .ckpt file found
    for file_path in ckpt_files:
        try:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")


def download_model_from_s3(local_file_name, bucket_name, s3_folder):
    """
    Downloads a model file from an S3 bucket to a local file.

    :param local_file_name: The local path where the model will be saved.
    :param bucket_name: The name of the S3 bucket.
    :param s3_folder: The S3 folder (prefix) from which to download the model.
    """
    # Create an S3 client
    s3_client = boto3.client('s3')

    # Construct the S3 file path
    s3_file_name = os.path.join(s3_folder, os.path.basename(local_file_name))
    print(f"s3 file name {s3_file_name} bucket name {bucket_name} local_file_name {local_file_name}")
    # Download the file
    print("+"*100)
    try:
        s3_client.download_file(bucket_name, s3_file_name, local_file_name)
        print(f"File {s3_file_name} downloaded from bucket {bucket_name} to {local_file_name}.")
    except Exception as e:
        print(f"Failed to download {s3_file_name} from S3: {e}")

# # Example usage
# # upload_file_to_s3('path/to/your/file.txt', 'your-s3-bucket-name')
# if __name__=='__main__':
#     upload_file_to_s3('/Users/chiragtagadiya/MyProjects/EMLO_V4_projects/DVC-pytorch-lightning-MLOps/checkpoints/bird_classification/best_model.ckpt', 'pytorch-model-emlov4')