"""Expand reader uploads into individual, pipeline-sized pages."""
from io import BytesIO
import os
from pathlib import Path
import re
import subprocess
import stat
import tempfile
import time
import zipfile
import zlib

from PIL import Image, ImageOps

from ..rendering.preparation import source_rgb
from .server import MAX_UPLOAD, RequestError

MAX_PAGES = 256
MAX_ZIP_DIRECTORY_BYTES = 16 * 1024**2


def require_space(location, size):
    space = os.statvfs(location)
    if space.f_bavail * space.f_frsize < size + 64 * 1024**2:
        raise RequestError("Insufficient disk space. Free some space and try again.", 507)


def webp_png(data):
    with Image.open(BytesIO(data), formats=("WEBP",)) as im:
        if im.format != "WEBP" or getattr(im, "n_frames", 1) != 1:
            raise ValueError("Choose a still WebP image.")
        if im.width * im.height > 4_000_000 or max(im.size) > 4096:
            raise ValueError("WebP exceeds the 4 megapixel or 4096 pixel side limit.")
        rgba = ImageOps.exif_transpose(im).convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        out = BytesIO()
        rgb.save(out, format="PNG")
        return out.getvalue(), rgb.size


def poppler(command, deadline):
    metadata = command[0] == "pdfinfo"
    timeout = min(45, deadline - time.monotonic())
    if timeout <= 0:
        raise ValueError("This PDF took too long to render.")
    # File-size rlimits apply to regular files, not pipes read into the parent.
    limit = 256 * 1024 if metadata else 32 * 1024**2
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(
                ["prlimit", "--as=1073741824", f"--fsize={limit}", "--", *command],
                stdout=output if metadata else subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout, env={**os.environ, "LC_ALL": "C"},
            )
            if result.returncode == 127:
                raise ValueError("Install Poppler on the server to import PDFs.")
            if result.returncode:
                raise ValueError("Cannot read this PDF. Check for damage or password protection.")
            if metadata:
                output.seek(0)
                return output.read(limit)
    except FileNotFoundError as exc:
        raise ValueError("PDF import needs Poppler and util-linux on the server.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("This PDF took too long to render.") from exc


def expand(name, data, directory, remaining):
    """Yield name, MIME, source bytes and size in document order."""
    path = data if isinstance(data, Path) else None
    if path:
        with path.open('rb') as stream:
            signature = stream.read(5)
    else:
        signature = data[:5]
    if signature == b"%PDF-" or name.lower().endswith(".pdf"):
        document = path or directory / "source.pdf"
        if path is None:
            require_space(directory, len(data))
            document.write_bytes(data)
        deadline = time.monotonic() + 300
        info = poppler(["pdfinfo", str(document)], deadline)
        match = re.search(rb"^Pages:\s+(\d+)\s*$", info, re.M)
        if not match or not 0 < int(match[1]) <= remaining:
            raise ValueError(f"An import can contain at most {MAX_PAGES} pages.")
        for page in range(1, int(match[1]) + 1):
            prefix = directory / "render"
            prefix.with_suffix(".png").unlink(missing_ok=True)
            require_space(directory, 32 * 1024**2)
            poppler(["pdftoppm", "-f", str(page), "-l", str(page), "-singlefile",
                     "-cropbox", "-scale-to", "2000", "-png", str(document), str(prefix)], deadline)
            image = prefix.with_suffix(".png").read_bytes()
            with Image.open(BytesIO(image), formats=("PNG",)) as im:
                yield f"{Path(name).stem[:180]}-{page:03}.png", "image/png", image, im.size
        return
    if Path(name).suffix.lower() in (".cbz", ".zip"):
        yield from archive_pages(data, remaining)
        return
    if path:
        with path.open('rb') as stream:
            data = stream.read(MAX_UPLOAD + 1)
    yield from image_page(name, data)


def path_order(path):
    """Compare directories and filenames component by component, with numeric runs."""
    def component_key(component):
        return tuple((1, len(part.lstrip("0")), part.lstrip("0")) if i % 2 else (0, part)
                     for i, part in enumerate(re.split(r"([0-9]+)", component.casefold())))
    return tuple(component_key(component) for component in path.split("/")), path


def archive_pages(data, remaining):
    try:
        with (data.open('rb') if isinstance(data, Path) else BytesIO(data)) as stream:
            # CPython's ZIP reader loads the entire central directory at open.
            # Bound that metadata separately from the streamed archive body.
            directory = zipfile._EndRecData(stream)
            if not directory or directory[zipfile._ECD_SIZE] > MAX_ZIP_DIRECTORY_BYTES:
                raise ValueError("Invalid archive or archive directory exceeds 16 MiB.")
            yield from zip_image_pages(stream, remaining)
    except (zipfile.BadZipFile, RuntimeError, EOFError, zlib.error) as exc:
        raise ValueError("Cannot read this archive. Check for damage, unsupported compression or encryption.") from exc


def zip_image_pages(stream, remaining):
    with zipfile.ZipFile(stream) as archive:
        images = []
        for entry in archive.infolist():
            name = entry.filename.replace(chr(92), "/")
            parts = name.split("/")
            if (entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16)
                    or any(not p or p.startswith(".") or p == "__MACOSX" for p in parts)
                    or Path(name).suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp")):
                continue
            images.append((name, entry))
            if len(images) > remaining:
                raise ValueError(f"An import can contain at most {MAX_PAGES} pages.")
            if entry.file_size > MAX_UPLOAD:
                raise ValueError(f"{name}: each image must be 32 MiB or smaller.")
        if not images:
            raise ValueError("Archive contains no PNG, JPEG, or WebP pages.")
        for name, entry in sorted(images, key=lambda item: path_order(item[0])):
            with archive.open(entry) as stream:
                content = stream.read(MAX_UPLOAD + 1)
            yield from image_page(name.rsplit("/", 1)[-1], content)


def image_page(name, data):
    if len(data) > MAX_UPLOAD:
        raise ValueError("Each image must be 32 MiB or smaller.")
    with Image.open(BytesIO(data), formats=("PNG", "JPEG", "WEBP")) as im:
        if im.format == "WEBP":
            _, size = webp_png(data)
            yield name, "image/webp", data, size
        elif im.format in ("PNG", "JPEG"):
            with source_rgb(im):
                yield name, Image.MIME[im.format], data, im.size
        else:
            raise ValueError("Choose a PNG, JPEG, WebP, or PDF file.")
