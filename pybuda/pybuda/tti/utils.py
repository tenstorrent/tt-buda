# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import hashlib
import os

def compute_file_checksum(file_path, chunk_size=8192):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)  # Read in chunks (default: 8KB)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def write_checksum_to_file(checksum, output_file_path):
    with open(output_file_path, 'w') as f:
        f.write(checksum)

def read_checksum_from_file(checksum_file_name):
    if not os.path.exists(checksum_file_name):
        return None
    with open(checksum_file_name, 'r') as f:
        checksum = f.read().strip()
    return checksum