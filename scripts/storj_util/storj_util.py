#!/usr/bin/env python3
"""
Script to compare remote Storj data with local data directory.
"""

import os
import sys
import json
import logging
import subprocess
import argparse

from pathlib import Path
from typing import Dict, List, Set, Tuple, NamedTuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FileInfo(NamedTuple):
    """Represents file information for comparison."""

    key: str
    size: int
    created: str
    kind: str


def setup_logger() -> logging.Logger:
    """Initialize and configure logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def get_bucket_path() -> str:
    """Get the DATA_FOLDER_BUCKET_PATH environment variable."""
    bucket_path = os.getenv("DATA_FOLDER_BUCKET_PATH")
    if not bucket_path:
        print("ERROR: DATA_FOLDER_BUCKET_PATH environment variable not found")
        sys.exit(1)
    return bucket_path


def get_remote_files(bucket_path: str, logger: logging.Logger) -> List[FileInfo]:
    """Get list of files from remote Storj bucket."""
    try:
        logger.info("Fetching remote file list from Storj...")
        cmd = f"uplink ls --recursive {bucket_path} --output json"

        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )

        remote_files = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                try:
                    file_data = json.loads(line)
                    if file_data.get("kind") == "OBJ":
                        remote_files.append(
                            FileInfo(
                                key=file_data["key"],
                                size=file_data["size"],
                                created=file_data["created"],
                                kind=file_data["kind"],
                            )
                        )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse JSON line: {line[:100]}... Error: {e}"
                    )

        logger.info(f"Found {len(remote_files)} remote files")
        return remote_files

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute uplink command: {e}")
        sys.exit(1)


def get_local_files(data_path: Path, logger: logging.Logger) -> Dict[str, FileInfo]:
    """Get list of files from local data directory."""
    local_files = {}

    if not data_path.exists():
        logger.warning(f"Local data directory {data_path} does not exist")
        return local_files

    logger.info("Scanning local data directory...")

    for file_path in data_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(data_path)
            key = str(relative_path).replace("\\", "/")

            try:
                stat = file_path.stat()
                local_files[key] = FileInfo(
                    key=key, size=stat.st_size, created=str(stat.st_ctime), kind="OBJ"
                )
            except OSError as e:
                logger.warning(f"Could not get file info for {file_path}: {e}")

    logger.info(f"Found {len(local_files)} local files")
    return local_files


def compare_files(
    remote_files: List[FileInfo], local_files: Dict[str, FileInfo]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Compare remote and local files."""
    remote_keys = {f.key for f in remote_files}
    local_keys = set(local_files.keys())

    only_remote = remote_keys - local_keys
    only_local = local_keys - remote_keys

    # Files in both but with different sizes
    different_sizes = set()
    remote_dict = {f.key: f for f in remote_files}

    for key in remote_keys & local_keys:
        if remote_dict[key].size != local_files[key].size:
            different_sizes.add(key)

    return only_remote, only_local, different_sizes


def download_file(
    bucket_path: str, key: str, local_path: Path, logger: logging.Logger
) -> bool:
    """Download a single file from Storj to local."""
    try:
        # Ensure the local directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Construct remote path - ensure proper formatting
        remote_path = f"{bucket_path}/{key}" if not bucket_path.endswith('/') else f"{bucket_path}{key}"
        cmd = f"uplink cp \"{remote_path}\" \"{local_path}\""

        logger.info(f"Downloading: {key}")
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )

        logger.info(f"✅ Downloaded: {key}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to download {key}: {e}")
        if e.stderr:
            logger.error(f"Error details: {e.stderr}")
        return False


def upload_file(
    bucket_path: str, key: str, local_path: Path, logger: logging.Logger
) -> bool:
    """Upload a single file from local to Storj."""
    try:
        # Ensure parent directories exist in remote bucket
        parent_dirs = Path(key).parent
        if str(parent_dirs) != '.':
            # Create parent directories if they don't exist
            for parent in parent_dirs.parents:
                if str(parent) != '.':
                    remote_dir = f"{bucket_path}/{parent}" if not bucket_path.endswith('/') else f"{bucket_path}{parent}"
                    try:
                        subprocess.run(
                            f"uplink mkdir \"{remote_dir}\"",
                            shell=True, capture_output=True, text=True, check=False
                        )
                    except Exception:
                        # Directory might already exist, continue
                        pass

        # Construct remote path - ensure proper formatting
        remote_path = f"{bucket_path}/{key}" if not bucket_path.endswith('/') else f"{bucket_path}{key}"
        cmd = f"uplink cp \"{local_path}\" \"{remote_path}\""

        logger.info(f"Uploading: {key}")
        _ = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

        logger.info(f"✅ Uploaded: {key}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to upload {key}: {e}")
        if e.stderr:
            logger.error(f"Error details: {e.stderr}")
        return False


def sync_files(
    bucket_path: str,
    local_data_path: Path,
    only_remote: Set[str],
    only_local: Set[str],
    different_sizes: Set[str],
    logger: logging.Logger,
    local_overwrites: bool = False,
) -> None:
    """Sync files between remote and local based on differences."""
    print("\n" + "=" * 60)
    print("SYNCING FILES")
    print("=" * 60)

    total_operations = len(only_remote) + len(only_local) + len(different_sizes)
    if total_operations == 0:
        print(" No files to sync - everything is already in sync!")
        return

    print(f"📊 Total operations to perform: {total_operations}")

    # Download files that are only remote
    if only_remote:
        print(f"\n📥 Downloading {len(only_remote)} files from remote...")
        success_count = 0
        for key in sorted(only_remote):
            local_path = local_data_path / key
            if download_file(bucket_path, key, local_path, logger):
                success_count += 1
        print(f"✅ Downloaded {success_count}/{len(only_remote)} files successfully")

    # Upload files that are only local
    if only_local:
        print(f"\n📤 Uploading {len(only_local)} files to remote...")
        success_count = 0
        for key in sorted(only_local):
            local_path = local_data_path / key
            if upload_file(bucket_path, key, local_path, logger):
                success_count += 1
        print(f"✅ Uploaded {success_count}/{len(only_local)} files successfully")

    # Handle files with different sizes
    if different_sizes:
        if local_overwrites:
            print(f"\n⚠️  Handling {len(different_sizes)} files with different sizes...")
            print("   (Uploading local version to overwrite remote)")
            success_count = 0
            for key in sorted(different_sizes):
                local_path = local_data_path / key
                if upload_file(bucket_path, key, local_path, logger):
                    success_count += 1
            print(
                f"✅ Updated {success_count}/{len(different_sizes)} files successfully"
            )
        else:
            print(f"\n⚠️  Handling {len(different_sizes)} files with different sizes...")
            print("   (Downloading remote version to overwrite local)")
            success_count = 0
            for key in sorted(different_sizes):
                local_path = local_data_path / key
                if download_file(bucket_path, key, local_path, logger):
                    success_count += 1
            print(
                f"✅ Updated {success_count}/{len(different_sizes)} files successfully"
            )

    print("\n🎉 Sync operation completed!")


def print_results(
    only_remote: Set[str],
    only_local: Set[str],
    different_sizes: Set[str],
    remote_files: List[FileInfo],
    local_files: Dict[str, FileInfo],
):
    """Print comparison results."""
    print("\n" + "=" * 60)
    print("STORJ DATA COMPARISON RESULTS")
    print("=" * 60)

    if only_remote:
        print(f"\n📁 Files only in remote Storj ({len(only_remote)}):")
        for key in sorted(only_remote):
            print(f"  + {key}")
    else:
        print("\n✅ No files only in remote Storj")

    if only_local:
        print(f"\n📁 Files only in local data ({len(only_local)}):")
        for key in sorted(only_local):
            print(f"  - {key}")
    else:
        print("\n✅ No files only in local data")

    if different_sizes:
        print(f"\n⚠️  Files with different sizes ({len(different_sizes)}):")
        remote_dict = {f.key: f for f in remote_files}
        for key in sorted(different_sizes):
            remote_size = remote_dict[key].size
            local_size = local_files[key].size
            print(f"  ≠ {key}")
            print(f"    Remote: {remote_size:,} bytes")
            print(f"    Local:  {local_size:,} bytes")
    else:
        print("\n✅ No files with different sizes")

    # Summary
    total_differences = len(only_remote) + len(only_local) + len(different_sizes)
    if total_differences == 0:
        print("\n🎉 Remote and local data are in sync!")
    else:
        print(f"\n📊 Summary: {total_differences} differences found")

    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare and sync Storj data with local directory"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync files after comparison (download missing files, upload new files)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Local data directory path (default: data)",
    )
    parser.add_argument(
        "--local-overwrites",
        action="store_true",
        help="When syncing, allow local files to overwrite remote files with different sizes",
    )
    args = parser.parse_args()

    logger = setup_logger()
    bucket_path = get_bucket_path()
    local_data_path = Path(args.data_dir)

    logger.info(f"Using bucket path: {bucket_path}")
    logger.info(f"Using local data path: {local_data_path}")

    # Get remote and local files
    remote_files = get_remote_files(bucket_path, logger)
    local_files = get_local_files(local_data_path, logger)

    # Compare files
    only_remote, only_local, different_sizes = compare_files(remote_files, local_files)

    # Print results
    print_results(only_remote, only_local, different_sizes, remote_files, local_files)

    # Sync files if requested
    if args.sync:
        sync_files(
            bucket_path,
            local_data_path,
            only_remote,
            only_local,
            different_sizes,
            logger,
            args.local_overwrites,
        )


if __name__ == "__main__":
    main()
