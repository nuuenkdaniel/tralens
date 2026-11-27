"""Exercise the packaged owner reader and the exact CPU OCR assets in CI."""
from contextlib import closing
from http.client import HTTPConnection
from io import BytesIO
import json
import os
from secrets import token_urlsafe
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageFont


port = int(os.environ["TRALENS_PORT"])
origin = os.environ["TRALENS_ORIGIN"]
host = urlsplit(origin).netloc if origin else f"127.0.0.1:{port}"
headers = {"Host": host, "Origin": origin or f"http://{host}"}


def request(method, path, body=None):
    with closing(HTTPConnection("127.0.0.1", port, timeout=5)) as connection:
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        return response.status, response.getheader("Set-Cookie"), response.read()


assert request("GET", "/healthz")[0] == 200
assert request("GET", "/")[0] == 200
assert request("GET", "/reader.js")[0] == 200
assert request("GET", "/verify-server.js")[0] == 200
status, _, body = request("GET", "/api/auth/discovery")
assert status == 200 and json.loads(body) == {"service": "tralens-reader", "version": 1}
assert request("GET", "/privacy")[0] == 200
assert request("GET", "/api/library")[0] == 401
headers["Content-Type"] = "application/json"

def sign_in(action, username, password):
    data = json.dumps({"username": username, "password": password})
    status, cookie, body = request("POST", "/api/auth/" + action, data)
    assert status == 204 and cookie and "HttpOnly" in cookie, (status, body)
    if origin:
        assert "; Secure" in cookie
    return cookie

owner_password = Path("/state/tmp/smoke-owner")
if not origin:
    owner_password.write_text(token_urlsafe(32))
cookie = sign_in("login" if origin else "setup", "smoke-owner", owner_password.read_text())
headers["Cookie"] = cookie.split(";", 1)[0]
assert request("GET", "/api/library")[0] == 200
status, _, body = request("GET", "/api/config")
assert status == 200
config = json.loads(body)
assert config["username"] == "smoke-owner"
assert config["render_enabled"] and config["worker_mode"] == "isolated_process"

if not origin:
    sys.exit()

restoring = len(sys.argv) > 1
status, _, body = request("GET", "/api/auth/settings")
assert status == 200 and json.loads(body)["registration"] == (not restoring)
# This credential belongs only to the temporary backup/restore fixture.
password_file = Path("/state/tmp/smoke-member")
if not restoring:
    password_file.write_text(token_urlsafe(32))
headers.pop("Cookie")
member = sign_in("login" if restoring else "register", "test-reader", password_file.read_text())
headers["Cookie"] = member.split(";", 1)[0]
status, _, body = request("GET", "/api/config")
assert status == 200 and json.loads(body)["username"] == "test-reader"
assert json.loads(body)["frontier_provider"] == "disabled"
assert request("GET", "/api/library")[0] == 200
headers["Cookie"] = cookie.split(";", 1)[0]

if restoring:
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

headers["Content-Type"] = "image/png"
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
