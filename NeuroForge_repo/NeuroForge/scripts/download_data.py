"""Download METR-LA (metr-la.h5 + sensor locations) into data/ and verify its SHA-256."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedtraffic.data import H5_SHA256, download, sha256  # noqa: E402

if __name__ == "__main__":
    h5, loc = download("data")
    digest = sha256(h5)
    print(f"{h5}: sha256 {digest}")
    if digest != H5_SHA256:
        print("WARNING: checksum differs from the copy used in our experiments")
        sys.exit(1)
    print("checksum OK")
