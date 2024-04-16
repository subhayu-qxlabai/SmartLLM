import boto3
from pathlib import Path
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AZURE_ACCESS_KEY,AZURE_CONNECTION_STR


class S3Client:
    def __init__(
        self,
        default_storage_account_name="smartllmstorage",
        storage_account_key=AZURE_ACCESS_KEY,
        container_name="llm-data",
        connection_string=AZURE_CONNECTION_STR,
        ) -> None:
        
        self.default_storage_account_name = default_storage_account_name
        self.storage_account_key = storage_account_key
        self.connection_string = connection_string
        self.container_name = container_name
        
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    def create_bucket(self, bucket_name: str = None) -> dict:
        bucket_name = bucket_name or self.default_bucket
        response = self.client.create_bucket(Bucket=bucket_name)
        return response

    def list_buckets(self) -> list[dict]:
        response = self.client.list_buckets()
        return response.get("Buckets", [])

    def list_objects(self, bucket_name: str = None) -> list[dict]:
        bucket_name = bucket_name or self.default_bucket
        response = self.client.list_objects(Bucket=bucket_name)
        return response.get("Contents", [])

    def download_file(
        self, object_key: str, local_parent: str = None, bucket_name: str = None, delete_remote = False,
    ):
        bucket_name = bucket_name or self.default_bucket
        local_parent: Path = Path(local_parent) if local_parent else Path(".")
        local_path = local_parent / object_key or Path(object_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket_name, object_key, local_path.as_posix())
        self.delete_file(object_key, bucket_name) if delete_remote else None
        return local_path
    
    def download_many(self, object_keys: list[str], local_parent: str = None, bucket_name: str = None, delete_remote = False):
        bucket_name = bucket_name or self.default_bucket
        local_parent: Path = Path(local_parent) if local_parent else Path(".")
        return [
            self.download_file(object_key, local_parent, bucket_name, delete_remote)
            for object_key in object_keys
        ]
        
            

    def upload_file(
        self,
        local_path: str | Path,
        object_key: str = None,
        bucket_name: str = None,
        metadata: dict = {},
        delete_local = True,
    ) -> str:
        
        file_name = os.path.split(local_path)[1]
        blob_client = self.blob_service_client.get_blob_client(container="llm-data", blob=file_name)
        
        local_path: Path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        with open(local_path.as_posix(),"rb") as data:
            blob_client.upload_blob(data)
            print(f"Uploaded {file_name}.")

        local_path.unlink() if delete_local else None

        # <StorageAccountName>.blob.core.windows.net/<ContainerName>/<FileNameWithExtension>
        return f"https://{self.default_storage_account_name}.blob.core.windows.net/{self.container_name}/{file_name}"

    def delete_file(self, object_key: str, bucket_name: str = None) -> dict:
        bucket_name = bucket_name or self.default_bucket
        response = self.client.delete_object(Bucket=bucket_name, Key=object_key)
        return response

    def get_presigned_url(
        self, object_key: str, bucket_name: str = None, expiration: int = 3600
    ) -> str:
        bucket_name = bucket_name or self.default_bucket
        response = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
            HttpMethod="GET",
        )
        return response
