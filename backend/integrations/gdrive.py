"""Google Drive API helper for authenticated downloads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import httpx


class GoogleDriveDownloader:
    """Download files from Google Drive using API authentication."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        service_account_path: Optional[str] = None,
    ) -> None:
        """
        Initialize with Google Drive API credentials.
        
        Options:
        1. API Key (for public files only):
           - Get from: https://console.cloud.google.com/apis/credentials
           - Set GOOGLE_DRIVE_API_KEY env var
        
        2. Service Account (for private files - recommended for corporate):
           - Create service account in Google Cloud Console
           - Download JSON key file
           - Share files with service account email
           - Set GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH env var
        """
        self.api_key = api_key or os.getenv("GOOGLE_DRIVE_API_KEY")
        self.service_account_path = service_account_path or os.getenv(
            "GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH"
        )
        self.base_url = "https://www.googleapis.com/drive/v3"
        self._service_account_credentials = None

    def _extract_file_id(self, url: str) -> Optional[str]:
        """Extract file ID from Google Drive URL."""
        if "drive.google.com" in url:
            # Handle /open?id= format
            if "/open?id=" in url:
                return url.split("/open?id=")[1].split("&")[0].split(",")[0]
            # Handle /file/d/ format
            elif "/file/d/" in url:
                return url.split("/file/d/")[1].split("/")[0]
        return None

    def _get_access_token(self) -> str:
        """Get OAuth2 access token using service account."""
        if not self.service_account_path:
            raise ValueError("Service account path not provided")

        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
        except ImportError:
            raise ImportError(
                "Google auth libraries not installed. "
                "Run: pip install google-auth google-api-python-client"
            )

        credentials_path = Path(self.service_account_path)
        if not credentials_path.exists():
            raise FileNotFoundError(
                f"Service account credentials file not found: {credentials_path}"
            )

        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path),
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        credentials.refresh(Request())
        return credentials.token

    async def download_file(self, url: str) -> bytes:
        """
        Download file from Google Drive using API.
        
        Uses service account if available, otherwise falls back to API key.
        """
        file_id = self._extract_file_id(url)
        if not file_id:
            raise ValueError(f"Could not extract file ID from URL: {url}")

        # Use service account if available (for private files)
        if self.service_account_path:
            try:
                access_token = self._get_access_token()
                headers = {"Authorization": f"Bearer {access_token}"}
            except Exception as e:
                raise ValueError(
                    f"Failed to get service account token: {e}. "
                    "Make sure service account JSON file is valid and files are shared with service account email."
                ) from e
        elif self.api_key:
            # API key only works for public files
            headers = {"Authorization": f"Bearer {self.api_key}"}
        else:
            raise ValueError(
                "No Google Drive credentials provided. "
                "Set GOOGLE_DRIVE_API_KEY or GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH in .env"
            )

        # Use Google Drive API to download
        download_url = f"{self.base_url}/files/{file_id}?alt=media"

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            response = await client.get(download_url, headers=headers)
            if response.status_code == 403:
                raise PermissionError(
                    f"Access denied to file {file_id}. "
                    "If using service account, make sure the file is shared with the service account email."
                )
            response.raise_for_status()
            return response.content

