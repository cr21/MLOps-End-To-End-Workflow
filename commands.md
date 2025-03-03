# DVC S3 Remote Setup Command
To set up DVC (Data Version Control) with an S3 remote for tracking your `data` folder, you can follow these steps. Below is the command to configure DVC with an S3 remote, along with a brief explanation.

### DVC S3 Remote Setup Command

```bash
# Initialize DVC in your project
dvc init

# Add your data folder to DVC tracking
dvc add data

# Set up the S3 remote storage
dvc remote add -d s3_data_store s3://your-bucket-name/path/to/data

# Configure AWS credentials (if not already set in your environment)
dvc remote modify s3_data_store access_key_id YOUR_ACCESS_KEY_ID
dvc remote modify s3_data_store secret_access_key YOUR_SECRET_ACCESS_KEY
dvc remote modify s3_data_store region YOUR_REGION
```

### Explanation of Commands

1. **`dvc init`**: Initializes a new DVC project in your current directory.
2. **`dvc add data`**: Tracks the `data` folder with DVC, creating a `.dvc` file for it.
3. **`dvc remote add -d s3_data_store s3://your-bucket-name/path/to/data`**: Adds a remote storage location (in this case, an S3 bucket) and sets it as the default remote.
4. **`dvc remote modify`**: Configures the access credentials and region for your S3 bucket.


Make sure to replace `your-bucket-name`, `YOUR_ACCESS_KEY_ID`, `YOUR_SECRET_ACCESS_KEY`, and `YOUR_REGION` with your actual S3 bucket details and AWS credentials. 


# Pushing data to S3
```bash
❯ dvc push
Collecting                                       |18.7k [00:01, 14.1kentry/s]
Pushing
17417 files pushed                                            
```                                                                        
                                                                             
     