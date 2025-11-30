#!/usr/bin/env python3
"""Run preprocessing only and save results locally for comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Get project root (parent of local_testing folder)
script_path = Path(__file__).resolve()
if script_path.parent.name == "local_testing":
    project_root = script_path.parent.parent
else:
    # Fallback: assume script is in project root
    project_root = script_path.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from backend.pipeline import PipelineConfig
from backend.pipeline.data_source import DataSourceResolver
from backend.preprocessing.audio_ops import AudioPreprocessor
from backend.integrations import GoogleDriveDownloader
from urllib.parse import urlparse
import httpx


async def download_audio(audio_url: str) -> bytes:
    """Download audio from URL (GDrive, HTTP, or local file)."""
    parsed = urlparse(audio_url)
    
    if "drive.google.com" in audio_url or "docs.google.com" in audio_url:
        downloader = GoogleDriveDownloader()
        return await downloader.download_file(audio_url)
    elif parsed.scheme in ("http", "https"):
        # HTTP/HTTPS
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(audio_url)
            response.raise_for_status()
            return response.content
    elif Path(audio_url).exists():
        # Local file path
        return Path(audio_url).read_bytes()
    else:
        raise ValueError(f"Unsupported audio source: {audio_url}. Supported: HTTP/HTTPS URLs, Google Drive links, or local file paths.")


async def preprocess_and_save_local(
    config_path: Path,
    output_dir: Path = None
) -> None:
    """Run preprocessing only and save original + processed audio locally."""
    
    # Resolve paths relative to project root
    script_path = Path(__file__).resolve()
    if script_path.parent.name == "local_testing":
        project_root = script_path.parent.parent
    else:
        project_root = script_path.parent
    
    # Resolve config path relative to project root if not absolute
    if not config_path.is_absolute():
        # Handle relative paths - if it starts with ../, resolve from current working directory
        # Otherwise, resolve from project root
        if str(config_path).startswith("../"):
            # Resolve from current working directory
            config_path = Path.cwd() / config_path
        else:
            # Resolve from project root
            config_path = project_root / config_path
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = project_root / "local_testing" / "preprocessing_output"
    elif not output_dir.is_absolute():
        # If relative path, resolve it properly
        output_str = str(output_dir)
        if output_str.startswith("local_testing/"):
            # Already has local_testing prefix, use as-is relative to project root
            output_dir = project_root / output_dir
        elif output_str.startswith("../"):
            # Resolve from current working directory
            output_dir = Path.cwd() / output_dir
        else:
            # For simple relative paths like "preprocessing_output",
            # prepend "local_testing/" to keep outputs organized
            if not output_str.startswith("local_testing/"):
                output_dir = project_root / "local_testing" / output_dir
            else:
                output_dir = project_root / output_dir
    
    # Load config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    client_config = config_data.get("clientConfig", {})
    dev_config = config_data.get("devConfig", {})
    
    # Load pipeline config (using resolved path)
    cfg = PipelineConfig.load(str(config_path))
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = output_dir / "original"
    processed_dir = output_dir / "processed"
    original_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir.absolute()}")
    print(f"  - Original audio: {original_dir}")
    print(f"  - Processed audio: {processed_dir}\n")
    
    # Load data source
    data_source_config = client_config.get("dataSource", {})
    source_type = data_source_config.get("type", "csv")
    
    if not data_source_config:
        source_type = "csv"
        data_source_config = {"csv": {"path": cfg.preprocessing.csv_path}}
    
    # Resolve CSV path relative to project root if it's a relative path
    if source_type == "csv" and "csv" in data_source_config:
        csv_path = data_source_config["csv"].get("path")
        if csv_path and not Path(csv_path).is_absolute():
            # Resolve relative to project root
            resolved_csv_path = project_root / csv_path
            data_source_config = data_source_config.copy()
            data_source_config["csv"] = data_source_config["csv"].copy()
            data_source_config["csv"]["path"] = str(resolved_csv_path)
    
    resolver = DataSourceResolver(
        source_type=source_type,
        source_config=data_source_config,
        dev_config=dev_config,
    )
    
    data_rows = resolver.load_data()
    print(f"Loaded {len(data_rows)} rows from {source_type}\n")
    
    # Process each row
    processed_count = 0
    for idx, row in enumerate(data_rows, start=1):
        print(f"Processing row {idx}/{len(data_rows)}...")
        
        # Extract audio URLs using the same logic as the original pipeline
        audio_sources = {}
        audio_index = 0
        
        for key, value in row.items():
            if not value or not isinstance(value, str):
                continue
            
            if not (value.startswith("http://") or value.startswith("https://")):
                continue
            
            key_lower = key.lower()
            is_audio_column = (
                "audio" in key_lower or 
                "upload" in key_lower or 
                "recording" in key_lower or
                "url" in key_lower
            )
            
            if is_audio_column:
                if audio_index == 0:
                    audio_sources["audio"] = value
                else:
                    audio_sources[f"audio_{audio_index}"] = value
                audio_index += 1
        
        if not audio_sources:
            print(f"  ⚠️  Skipping row {idx}: No audio URL found")
            continue
        
        # Process each audio source found in the row
        for channel_name, audio_url in audio_sources.items():
            try:
                # Download original audio
                print(f"  📥 Downloading [{channel_name}]: {audio_url}")
                raw_bytes = await download_audio(audio_url)
                
                # Save original
                original_filename = f"row_{idx}_{channel_name}_original.wav"
                original_path = original_dir / original_filename
                with open(original_path, "wb") as f:
                    f.write(raw_bytes)
                print(f"  💾 Saved original: {original_path.name} ({len(raw_bytes) / 1024:.1f} KB)")
                
                # Process audio
                if cfg.preprocessing.clear_audio:
                    audio_processor = AudioPreprocessor(
                        clear_config=cfg.preprocessing.clear_audio,
                        enable_noise_check=False,
                        enable_segmentation=False,
                    )
                    
                    processed_bytes, preprocessing_meta = await audio_processor.preprocess_audio(
                        raw_bytes,
                        filename=Path(audio_url).name,
                    )
                    
                    # Save processed
                    processed_filename = f"row_{idx}_{channel_name}_processed.wav"
                    processed_path = processed_dir / processed_filename
                    with open(processed_path, "wb") as f:
                        f.write(processed_bytes)
                    print(f"  ✅ Saved processed: {processed_path.name} ({len(processed_bytes) / 1024:.1f} KB)")
                    
                    # Save metadata
                    meta_filename = f"row_{idx}_{channel_name}_metadata.json"
                    meta_path = output_dir / meta_filename
                    with open(meta_path, "w") as f:
                        json.dump({
                            "row_index": idx,
                            "channel": channel_name,
                            "original_url": audio_url,
                            "preprocessing_applied": preprocessing_meta.get("preprocessing_applied", []),
                            "original_size_bytes": preprocessing_meta.get("original_size_bytes"),
                            "processed_size_bytes": preprocessing_meta.get("processed_size_bytes"),
                        }, f, indent=2)
                    
                    processed_count += 1
                    applied = preprocessing_meta.get('preprocessing_applied', [])
                    if applied:
                        print(f"  📊 Applied: {', '.join(applied)}\n")
                    else:
                        print(f"  📊 No preprocessing applied\n")
                else:
                    print(f"  ⚠️  clearAudio not enabled, skipping processing\n")
            
            except Exception as e:
                print(f"  ❌ Error processing row {idx} [{channel_name}]: {e}\n")
                import traceback
                traceback.print_exc()
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Processed: {processed_count}/{len(data_rows)} rows")
    print(f"   Output folder: {output_dir.absolute()}")
    print(f"\n   Compare files:")
    print(f"   - Original: {original_dir}")
    print(f"   - Processed: {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preprocessing only and save results locally for comparison."
    )
    parser.add_argument(
        "--config",
        default="config/modules.yaml",
        help="Path to modules YAML config (default: config/modules.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for original and processed audio (default: local_testing/preprocessing_output)",
    )
    args = parser.parse_args()
    
    asyncio.run(preprocess_and_save_local(
        config_path=Path(args.config),
        output_dir=Path(args.output) if args.output else None,
    ))


if __name__ == "__main__":
    main()

