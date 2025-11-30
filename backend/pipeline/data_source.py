"""Data source resolver for CSV, DynamoDB, and API sources."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

import httpx


class DataSourceResolver:
    """Resolves data from different sources based on config."""

    def __init__(
        self,
        source_type: str,
        source_config: Dict[str, Any],
        dev_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.source_type = source_type
        self.source_config = source_config
        self.dev_config = dev_config or {}
        self._dynamodb_client: Optional[Any] = None

    def load_data(self) -> List[Dict[str, str]]:
        """Load data based on source type."""
        source_type = self.source_type.lower() if self.source_type else "csv"
        if source_type == "csv":
            return self._load_csv()
        elif source_type == "dynamodb":
            return self._load_dynamodb()
        elif source_type == "api":
            return self._load_api()
        else:
            raise ValueError(f"Unsupported data source type: {self.source_type}. Supported: csv, dynamodb, api")

    def _load_csv(self) -> List[Dict[str, str]]:
        """Load data from CSV file."""
        csv_config = self.source_config.get("csv", {})
        csv_path = csv_config.get("path")
        
        # Fallback to dev config if path not in client config
        if not csv_path:
            dev_ingestion = self.dev_config.get("ingestion", {})
            csv_path = dev_ingestion.get("csv", {}).get("localFallbackPath")
        
        if not csv_path:
            raise ValueError("CSV path not specified in clientConfig.dataSource.csv.path or devConfig.ingestion.csv.localFallbackPath")
        
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        delimiter = csv_config.get("delimiter", ",")
        
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            return []

        header = rows[0]
        deduped_header: List[str] = []
        seen: Dict[str, int] = {}

        for idx, col in enumerate(header):
            clean_name = (col or "").strip()
            if not clean_name:
                clean_name = f"column_{idx}"

            count = seen.get(clean_name, 0)
            if count > 0:
                deduped_name = f"{clean_name}__{count+1}"
            else:
                deduped_name = clean_name
            seen[clean_name] = count + 1
            deduped_header.append(deduped_name)

        records: List[Dict[str, str]] = []
        for row in rows[1:]:
            # Pad or trim row to match header length
            if len(row) < len(deduped_header):
                row = row + [""] * (len(deduped_header) - len(row))
            elif len(row) > len(deduped_header):
                row = row[: len(deduped_header)]
            records.append(dict(zip(deduped_header, row)))

        return records

    def _load_dynamodb(self) -> List[Dict[str, str]]:
        """Load data from DynamoDB table."""
        dynamodb_config = self.source_config.get("dynamodb", {})
        dev_ingestion = self.dev_config.get("ingestion", {})
        dev_dynamodb = dev_ingestion.get("dynamodb", {})
        
        table_name = dynamodb_config.get("tableName") or dev_dynamodb.get("tableName")
        region = dynamodb_config.get("region") or dev_dynamodb.get("region", "us-east-1")
        filter_expression = dynamodb_config.get("filterExpression")
        projection_fields = dynamodb_config.get("projectionFields", [])
        batch_size = dev_dynamodb.get("batchSize", 50)
        
        if not table_name:
            raise ValueError("DynamoDB table name not specified in clientConfig.dataSource.dynamodb.tableName or devConfig.ingestion.dynamodb.tableName")

        if not self._dynamodb_client:
            self._dynamodb_client = boto3.client("dynamodb", region_name=region)

        records: List[Dict[str, str]] = []
        scan_kwargs: Dict[str, Any] = {
            "TableName": table_name,
            "Limit": batch_size,
        }
        
        if projection_fields:
            scan_kwargs["ProjectionExpression"] = ", ".join(projection_fields)
        
        if filter_expression:
            # Simple expression value substitution (for basic cases)
            # For complex expressions, client should provide ExpressionAttributeValues
            scan_kwargs["FilterExpression"] = filter_expression

        try:
            while True:
                response = self._dynamodb_client.scan(**scan_kwargs)
                
                for item in response.get("Items", []):
                    # Convert DynamoDB item to flat dict
                    record: Dict[str, str] = {}
                    for key, value_obj in item.items():
                        # Handle DynamoDB attribute value types
                        if "S" in value_obj:  # String
                            record[key] = value_obj["S"]
                        elif "N" in value_obj:  # Number
                            record[key] = value_obj["N"]
                        elif "BOOL" in value_obj:  # Boolean
                            record[key] = str(value_obj["BOOL"])
                        else:
                            record[key] = str(value_obj)
                    records.append(record)
                
                if "LastEvaluatedKey" not in response:
                    break
                scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
                
        except ClientError as e:
            raise RuntimeError(f"Failed to scan DynamoDB table {table_name}: {e}")

        return records

    def _load_api(self) -> List[Dict[str, str]]:
        """Load data from API endpoint."""
        api_config = self.source_config.get("api", {})
        endpoint = api_config.get("endpoint")
        
        if not endpoint:
            raise ValueError("API endpoint not specified in clientConfig.dataSource.api.endpoint")

        auth_type = api_config.get("authType", "none")
        headers = api_config.get("headers", {})
        
        # Add auth headers
        if auth_type == "basic":
            username = os.getenv("API_USERNAME")
            password = os.getenv("API_PASSWORD")
            if not username or not password:
                raise ValueError("API_USERNAME and API_PASSWORD env vars required for basic auth")
            import base64
            auth_str = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {auth_str}"
        elif auth_type == "bearer":
            token = os.getenv("API_TOKEN") or os.getenv("API_BEARER_TOKEN")
            if not token:
                raise ValueError("API_TOKEN or API_BEARER_TOKEN env var required for bearer auth")
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = httpx.get(endpoint, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                # Direct list of records
                return [self._normalize_record(record) for record in data]
            elif isinstance(data, dict):
                # Check common keys
                if "data" in data:
                    records = data["data"]
                elif "items" in data:
                    records = data["items"]
                elif "results" in data:
                    records = data["results"]
                else:
                    # Single record
                    return [self._normalize_record(data)]
                
                if isinstance(records, list):
                    return [self._normalize_record(record) for record in records]
                else:
                    raise ValueError(f"Unexpected API response format: {type(records)}")
            else:
                raise ValueError(f"Unexpected API response type: {type(data)}")
                
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"API request failed with status {e.response.status_code}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data from API: {e}")

    def _normalize_record(self, record: Any) -> Dict[str, str]:
        """Normalize a record to Dict[str, str] format."""
        if isinstance(record, dict):
            return {str(k): str(v) if v is not None else "" for k, v in record.items()}
        else:
            raise ValueError(f"Cannot normalize record of type {type(record)}")


def validate_audio_location(
    audio_url: str,
    audio_preferences: Dict[str, Any],
    runtime_checks: Dict[str, Any],
) -> bool:
    """Validate if audio URL matches allowed locations."""
    from urllib.parse import urlparse
    
    if not runtime_checks.get("requireKnownSource", True):
        return True
    
    parsed = urlparse(audio_url)
    
    # Check S3
    if parsed.scheme == "s3" or (parsed.netloc and "s3" in parsed.netloc.lower()):
        return "s3" in audio_preferences.get("acceptedLocations", [])
    
    # Check Google Drive
    if "drive.google.com" in parsed.netloc or "docs.google.com" in parsed.netloc:
        return "gdrive" in audio_preferences.get("acceptedLocations", [])
    
    # Check if external sources are allowed
    if runtime_checks.get("allowExternalAudio", False):
        external_sources = runtime_checks.get("externalSources", [])
        if "gdrive" in external_sources and ("drive.google.com" in parsed.netloc or "docs.google.com" in parsed.netloc):
            return True
    
    # HTTP/HTTPS URLs - check if they're in accepted locations
    if parsed.scheme in ("http", "https"):
        # If no specific restrictions, allow HTTP/HTTPS
        if not runtime_checks.get("requireKnownSource", True):
            return True
        # Otherwise, only allow if explicitly listed
        return "http" in audio_preferences.get("acceptedLocations", []) or "https" in audio_preferences.get("acceptedLocations", [])
    
    # Default: reject unknown sources
    return False

