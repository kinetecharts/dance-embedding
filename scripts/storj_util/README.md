# Storj Data Comparison and Sync Tool

A Python script for comparing and synchronizing files between a local data directory and a remote Storj bucket using the Uplink CLI.

## Overview

This tool helps you:

- **Compare** files between your local directory and remote Storj bucket
- **Identify** differences (missing files, extra files, size mismatches)
- **Synchronize** data by downloading missing files or uploading new ones
- **Resolve conflicts** when files exist in both locations but have different sizes

## Features

- 🔍 **Comprehensive Comparison**: Detects files that exist only remotely, only locally, or have different sizes
- 📊 **Detailed Reporting**: Shows file counts, sizes, and specific differences with formatted output
- 🔄 **Bidirectional Sync**: Can download from remote or upload to remote
- ⚙️ **Conflict Resolution**: Choose whether local or remote files take precedence
- 📝 **Detailed Logging**: Comprehensive logging with timestamps and status indicators
- 🛡️ **Error Handling**: Graceful handling of network issues and file access problems
- 📁 **Directory Structure Preservation**: Maintains folder hierarchy during sync operations
- ✅ **Success Tracking**: Reports success/failure counts for each operation type

## Prerequisites

### 1. Python Dependencies

```bash
pip install python-dotenv
```

**Note**: The script uses standard Python libraries (`subprocess`, `argparse`, `pathlib`, `typing`, `logging`, `json`, `os`, `sys`) which are included with Python.

### 2. Storj Uplink CLI

Install the Storj Uplink CLI and configure it with your access credentials:

- Download from: <https://docs.storj.io/dcs/getting-started/quickstart-uplink-cli/>
- Follow the setup guide to configure your access
- Ensure `uplink` command is available in your PATH

### 3. Environment Configuration

Create a `.env` file in the same directory as the script with:

```env
DATA_FOLDER_BUCKET_PATH=your-bucket-name/optional-prefix
```

**Note**: The bucket path can include a prefix to sync only a specific folder within your bucket.

## Usage

### Basic Comparison (No Sync)

```bash
python diff_storj_data.py
```
This will compare your local `data` directory with the remote Storj bucket and show differences without making any changes.

### Compare with Custom Data Directory

```bash
python diff_storj_data.py --data-dir /path/to/your/data
```

### Sync Files (Download Missing, Upload New)

```bash
python diff_storj_data.py --sync
```

### Sync with Local Files Taking Precedence

```bash
python diff_storj_data.py --sync --local-overwrites
```

### Complete Example

```bash
# Compare and sync, allowing local files to overwrite remote conflicts
python diff_storj_data.py --sync --local-overwrites --data-dir ./my-data
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sync` | Sync files after comparison | False |
| `--data-dir` | Local data directory path | `data` |
| `--local-overwrites` | Allow local files to overwrite remote files with different sizes | False |

## How It Works

### 1. File Discovery

- **Remote**: Uses `uplink ls --recursive --output json` to get file list from Storj, parsing JSON output for file metadata
- **Local**: Scans the specified directory recursively for files using `pathlib.Path.rglob()`

### 2. Comparison Logic

The script identifies three types of differences:
- **Only Remote**: Files that exist in Storj but not locally
- **Only Local**: Files that exist locally but not in Storj  
- **Different Sizes**: Files that exist in both places but have different file sizes

### 3. Synchronization Strategy

When `--sync` is used:
- **Only Remote files**: Downloaded to local directory (creates parent directories as needed)
- **Only Local files**: Uploaded to remote bucket
- **Different Sizes**: 
  - Without `--local-overwrites`: Remote version overwrites local
  - With `--local-overwrites`: Local version overwrites remote

### 4. Error Handling

- Individual file operations are tracked for success/failure
- Failed operations are logged with detailed error messages
- The script continues processing other files even if some fail
- Summary reports show success counts for each operation type

## File Information Structure

The script uses a `FileInfo` named tuple to track file metadata:

```python
FileInfo(
    key=str,      # File path relative to root
    size=int,     # File size in bytes
    created=str,  # Creation timestamp (Unix timestamp as string)
    kind=str      # Always "OBJ" for objects
)
```

## Error Scenarios

The script handles various error conditions:

1. **Missing Environment Variable**: Exits with error if `DATA_FOLDER_BUCKET_PATH` is not set
2. **Uplink Command Failure**: Logs error and exits if Storj CLI commands fail
3. **Local Directory Missing**: Warns but continues if local directory doesn't exist
4. **File Access Errors**: Logs warnings for files that can't be read
5. **JSON Parsing Errors**: Warns for malformed JSON from Storj CLI output
6. **Individual File Operations**: Continues processing other files if one fails

## Best Practices

1. **Test First**: Always run without `--sync` first to review differences
2. **Backup Important Data**: Ensure you have backups before syncing
3. **Use Specific Bucket Paths**: Use prefixes to sync only relevant folders
4. **Monitor Logs**: Check the detailed logs for any failed operations
5. **Verify Results**: Run comparison again after sync to confirm success

## Troubleshooting

### Common Issues

1. **"uplink command not found"**
   - Ensure Storj Uplink CLI is installed and in your PATH
   - Verify installation with `uplink --version`

2. **"DATA_FOLDER_BUCKET_PATH environment variable not found"**
   - Create a `.env` file in the script directory
   - Set the correct bucket path: `DATA_FOLDER_BUCKET_PATH=your-bucket/path`

3. **Permission errors during sync**
   - Check file permissions in local directory
   - Verify Storj access credentials are properly configured

4. **Network timeouts**
   - The script will log individual file failures
   - Re-run sync to retry failed operations

### Getting Help

- Check the detailed logs for specific error messages
- Verify your Storj configuration with `uplink ls`
- Ensure your local directory has proper read/write permissions
