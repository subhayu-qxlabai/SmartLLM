import boto3
from pathlib import Path
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL


class S3Client:
    def __init__(
        self,
        default_bucket_name="smartllm",
        endpoint_url=AWS_ENDPOINT_URL,
        access_key=AWS_ACCESS_KEY_ID,
        secret_key=AWS_SECRET_ACCESS_KEY,
    ) -> None:
        self.default_bucket = default_bucket_name
        self.endpoint_url = endpoint_url
        self.https_endpoint_url = (
            f"https://{endpoint_url}"
            if not endpoint_url.startswith("https://")
            else endpoint_url
        )
        self.client = boto3.client(
            "s3",
            endpoint_url=self.https_endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

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
        self, object_key: str, local_parent: str = None, bucket_name: str = None
    ) -> dict:
        bucket_name = bucket_name or self.default_bucket
        local_parent: Path = Path(local_parent) if local_parent else Path(".")
        local_path = local_parent / object_key or Path(object_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket_name, object_key, local_path.as_posix())
        return local_path

    def upload_file(
        self, local_path: str, object_key: str = None, bucket_name: str = None
    ) -> dict:
        bucket_name = bucket_name or self.default_bucket
        local_path: Path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        object_key = local_path.as_posix() if not object_key else object_key
        self.client.upload_file(local_path.as_posix(), bucket_name, object_key)
        return f"https://{bucket_name}.{self.endpoint_url}/{object_key}"

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
