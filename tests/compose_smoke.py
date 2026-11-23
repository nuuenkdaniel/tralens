"""Exercise the packaged owner reader and the exact CPU OCR assets in CI."""
from contextlib import closing
from http.client import HTTPConnection
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageFont


port = int(os.environ["TRALENS_PORT"])
origin = os.environ["TRALENS_ORIGIN"]
headers = {"Host": urlsplit(origin).netloc, "Origin": origin}


def request(method, path, body=None):
    with closing(HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        return response.status, response.getheader("Set-Cookie"), response.read()


assert request("GET", "/healthz")[0] == 200
assert request("GET", "/")[0] == 200
assert request("GET", "/reader.js")[0] == 200
assert request("GET", "/privacy")[0] == 200
assert request("GET", "/api/library")[0] == 401
key = Path("/config/owner.key").read_bytes().strip()
status, cookie, _ = request("POST", "/api/owner/login", key)
assert status == 204 and cookie
headers["Cookie"] = cookie.split(";", 1)[0]
assert request("GET", "/api/library")[0] == 200
status, _, body = request("GET", "/api/config")
assert status == 200
config = json.loads(body)
assert config["render_enabled"] and config["worker_mode"] == "isolated_process"

if len(sys.argv) > 1:
    status, _, body = request("GET", f"/api/jobs/{sys.argv[1]}")
    assert status == 200 and json.loads(body)["status"] == "complete", body
    print("Restored the completed OCR job.")
    sys.exit()

page = Image.new("RGB", (640, 240), "white")
manifest = json.loads(Path("/app/fonts.json").read_text())
face = ImageFont.truetype(manifest["fonts"][0]["path"], 74)
ImageDraw.Draw(page).text((65, 65), "Tralens", font=face, fill="black")
png = BytesIO()
page.save(png, format="PNG")

headers["X-Tralens-Token"] = config["token"]
headers["X-Tralens-Options"] = '{"mode":"ocr"}'
status, _, body = request("POST", "/api/jobs", png.getvalue())
assert status == 202, body
job = json.loads(body)["id"]
deadline = time.monotonic() + 180
while True:
    status, _, body = request("GET", f"/api/jobs/{job}")
    assert status == 200, body
    result = json.loads(body)
    if result["status"] != "running":
        break
    if time.monotonic() >= deadline:
        raise TimeoutError(f"CPU OCR did not finish: {result}")
    time.sleep(1)
assert result["status"] == "complete", result
assert any("tralens" in row["text"].lower() for row in result["detections"]), result
print(json.dumps({"job": job}))
