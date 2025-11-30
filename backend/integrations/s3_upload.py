"""S3 upload functionality for final data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import asyncio
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import ClientError


class S3Uploader:
    """Upload files to S3."""

    def __init__(
        self,
        bucket_name: str,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region: str = "ap-south-1",
    ) -> None:
        """
        Initialize S3 uploader.

        Args:
            bucket_name: S3 bucket name
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            region: AWS region
        """
        self.bucket_name = bucket_name
        self.region = region

        # Get credentials from args or environment
        self.access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        if not self.access_key_id or not self.secret_access_key:
            raise ValueError(
                "AWS credentials not provided. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region,
        )

        # Thread pool for async-style uploads using run_in_executor
        workers = int(os.getenv("S3_UPLOAD_WORKERS", "8"))
        self._executor = ThreadPoolExecutor(max_workers=workers)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool executor."""
        if self._executor:
            self._executor.shutdown(wait=wait)
    
    def __del__(self) -> None:
        """Cleanup thread pool on deletion."""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=False)

    def upload_file(
        self,
        file_path: str | Path,
        s3_key: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload file to S3.

        Args:
            file_path: Local file path
            s3_key: S3 object key (path in bucket)
            metadata: Optional metadata dict

        Returns:
            Public HTTPS URL of uploaded file (bucket must allow public read)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extra_args = {}
        if metadata:
            extra_args["Metadata"] = {str(k): str(v) for k, v in metadata.items()}

        try:
            self.s3_client.upload_file(
                str(file_path),
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args,
            )
            
            return self._build_public_url(s3_key)
        except ClientError as e:
            raise RuntimeError(f"Failed to upload to S3: {e}") from e

    def upload_bytes(
        self,
        file_bytes: bytes,
        s3_key: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload file bytes to S3.

        Args:
            file_bytes: File content as bytes
            s3_key: S3 object key (path in bucket)
            metadata: Optional metadata dict

        Returns:
            Public HTTPS URL of uploaded file (bucket must allow public read)
        """
        extra_args = {}
        if metadata:
            extra_args["Metadata"] = {str(k): str(v) for k, v in metadata.items()}

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_bytes,
                **extra_args,
            )
            
            return self._build_public_url(s3_key)
        except ClientError as e:
            raise RuntimeError(f"Failed to upload to S3: {e}") from e

    async def upload_bytes_async(
        self,
        file_bytes: bytes,
        s3_key: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Async wrapper around upload_bytes using a thread pool.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self.upload_bytes,
            file_bytes,
            s3_key,
            metadata,
        )

    def upload_directory(
        self,
        local_dir: str | Path,
        s3_prefix: str,
        metadata: Optional[dict] = None,
    ) -> List[str]:
        """
        Upload entire directory to S3.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix (folder path)
            metadata: Optional metadata dict

        Returns:
            List of uploaded S3 URLs
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise ValueError(f"Not a directory: {local_dir}")

        uploaded_urls = []
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")
                uploaded_urls.append(self.upload_file(file_path, s3_key, metadata))

        return uploaded_urls

    def download_bytes(self, s3_key: str) -> bytes:
        """Download object bytes from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response["Body"].read()
        except ClientError as e:
            raise RuntimeError(f"Failed to download from S3: {e}") from e

    def _build_public_url(self, s3_key: str) -> str:
        """
        Construct a public HTTPS URL for the given S3 key using the default bucket.
        """
        return self._build_public_url_for_bucket(self.bucket_name, s3_key)

    def _build_public_url_for_bucket(self, bucket_name: str, s3_key: str) -> str:
        """
        Construct a public HTTPS URL for the given S3 key and bucket.
        """
        sanitized_key = s3_key.lstrip("/")
        return f"https://{bucket_name}.s3.{self.region}.amazonaws.com/{sanitized_key}"

    def upload_file_to_bucket(
        self,
        bucket_name: str,
        file_path: str | Path,
        s3_key: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload file to a specific bucket, overriding the default bucket_name.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extra_args = {}
        if metadata:
            extra_args["Metadata"] = {str(k): str(v) for k, v in metadata.items()}

        try:
            self.s3_client.upload_file(
                str(file_path),
                bucket_name,
                s3_key,
                ExtraArgs=extra_args,
            )
            return self._build_public_url_for_bucket(bucket_name, s3_key)
        except ClientError as e:
            raise RuntimeError(f"Failed to upload to S3 bucket {bucket_name}: {e}") from e

    def upload_bytes_to_bucket(
        self,
        bucket_name: str,
        file_bytes: bytes,
        s3_key: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Upload bytes to a specific bucket, overriding the default bucket_name.
        """
        extra_args = {}
        if metadata:
            extra_args["Metadata"] = {str(k): str(v) for k, v in metadata.items()}

        try:
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=file_bytes,
                **extra_args,
            )
            return self._build_public_url_for_bucket(bucket_name, s3_key)
        except ClientError as e:
            raise RuntimeError(f"Failed to upload to S3 bucket {bucket_name}: {e}") from e

