"""Trusted HTTPS origin and bootstrap credential for the self-hosted operator."""
import hashlib
from urllib.parse import urlsplit


class Owner:
    def __init__(self, origin, key_path):
        url = urlsplit(origin)
        if url.scheme != 'https' or not url.hostname or url.username is not None or origin != 'https://' + url.netloc:
            raise ValueError('Set the exact public HTTPS origin without a trailing slash.')
        if key_path.stat().st_mode & 0o077:
            raise ValueError('The owner key file must be private (mode 0600).')
        key = key_path.read_bytes().strip()
        if not 32 <= len(key) <= 256 or not key.isascii():
            raise ValueError('Use a random owner key of 32–256 ASCII characters.')
        self.digest = hashlib.sha256(key).digest()
        self.origin = origin
        self.host = url.netloc
