"""Bind the Debian-packaged dialogue face to its installed license text."""
from hashlib import sha256
import json
from pathlib import Path


def entry(identity,role,weight,font,license_file,license_name):
    return dict(id=identity,role=role,weight=weight,path=str(font),
                sha256=sha256(font.read_bytes()).hexdigest(),license=license_name,
                license_path=str(license_file),license_sha256=sha256(license_file.read_bytes()).hexdigest())


comic = Path("/usr/share/fonts/opentype/comic-neue")
comic_license = Path("/usr/share/doc/fonts-comic-neue/copyright")
fonts = [
    entry("dejavu-sans","dialogue","regular",Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
          Path("/usr/share/doc/fonts-dejavu-core/copyright"),"Bitstream Vera and Arev font licenses"),
    entry("comic-neue-bold-italic","shout","bold",comic/"ComicNeue-BoldItalic.otf",comic_license,"SIL OFL 1.1"),
    entry("comic-neue-regular","handwriting","regular",comic/"ComicNeue-Regular.otf",comic_license,"SIL OFL 1.1"),
]
Path("/app/fonts.json").write_text(json.dumps({"schema_version":1,"default_font_id":"dejavu-sans","fonts":fonts})+"\n")
