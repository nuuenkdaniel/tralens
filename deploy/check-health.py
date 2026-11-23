"""Check that the reader responds after opening its library and validating assets."""
from contextlib import closing
from http.client import HTTPConnection
import os
from urllib.parse import urlsplit


port = int(os.environ.get("TRALENS_PORT", "7871"))
origin = os.environ.get("TRALENS_ORIGIN", "")
headers = {"Host": urlsplit(origin).netloc, "Origin": origin} if origin else {}
with closing(HTTPConnection("127.0.0.1", port, timeout=3)) as connection:
    connection.request("GET", "/healthz", headers=headers)
    if connection.getresponse().status != 200:
        raise SystemExit("Tralens is not ready.")
