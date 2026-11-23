"""Bind the Debian-packaged dialogue face to its installed license text."""
from hashlib import sha256
import json
from pathlib import Path


font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
license_file = Path("/usr/share/doc/fonts-dejavu-core/copyright")
manifest = {
    "schema_version": 1,
    "fonts": [{
        "id": "dejavu-sans",
        "role": "dialogue",
        "weight": "regular",
        "path": str(font),
        "sha256": sha256(font.read_bytes()).hexdigest(),
        "license": "Bitstream Vera and Arev font licenses",
        "license_path": str(license_file),
        "license_sha256": sha256(license_file.read_bytes()).hexdigest(),
    }],
}
Path("/app/fonts.json").write_text(json.dumps(manifest) + "\n")
