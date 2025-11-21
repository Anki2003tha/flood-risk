"""Helper utilities to download a small Sentinel sample (templates only).
This module intentionally prints instructions and provides a small boto3 helper
for public S3 buckets. Do not run automatic massive downloads without
checking footprint and disk space.
"""

from pathlib import Path


def print_instructions():
    print("Sentinel-2 sample download instructions:\n")
    print("1) Use the Copernicus Open Access Hub and the 'sentinelsat' Python package to query and download specific scenes.\n")
    print("   Example (template):\n   from sentinelsat import SentinelAPI\n   api = SentinelAPI('user','password','https://scihub.copernicus.eu/dhus')\n")
    print("2) Many public tiles are available via AWS. Use the AWS CLI for fast downloads:\n   aws s3 cp \"s3://<bucket>/<tile>.zip\" . --no-sign-request\n")
    print("3) Small helper: download a public S3 object using boto3 (if installed).\n   See download_s3_object(s3_uri, out_dir) in this module.\n")


def download_s3_object(s3_uri, out_dir='.'):
    """Download an S3 object to out_dir. Returns True on success.
    s3_uri example: 's3://bucket/path/to/file.zip'
    """
    try:
        import boto3
        from urllib.parse import urlparse
    except Exception as e:
        print('boto3 not available. Install boto3 to use this helper:', e)
        return False
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(key).name
    s3 = boto3.client('s3')
    print(f'Downloading s3://{bucket}/{key} -> {out_path}')
    try:
        s3.download_file(bucket, key, str(out_path))
        print('Download complete')
        return True
    except Exception as e:
        print('Download failed:', e)
        return False


if __name__ == '__main__':
    print_instructions()
