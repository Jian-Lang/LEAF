import hashlib
import os
import shutil


def calculate_md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()
