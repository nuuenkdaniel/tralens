"""Stream reader form uploads to per-request staging files."""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header

from .import_pages import MAX_PAGES
from .server import RequestError

MAX_IMPORT_BYTES = 4 * 1024**3
MAX_METADATA_BYTES = 8192
MAX_PART_HEADER_BYTES = 16384


@contextmanager
def read_import(stream, length, content_type):
    mime, options = parse_options_header(content_type)
    boundary = options.get(b'boundary', b'')
    if mime != b'multipart/form-data' or not 0 < len(boundary) <= 70:
        raise RequestError('Multipart page upload required.')
    with TemporaryDirectory(prefix='tralens-upload-') as directory:
        upload = ImportParts(Path(directory))
        parser = MultipartParser(boundary, {
            'on_part_begin': upload.part_begin,
            'on_header_field': upload.header_field,
            'on_header_value': upload.header_value,
            'on_header_end': upload.header_end,
            'on_headers_finished': upload.headers_finished,
            'on_part_data': upload.part_data,
            'on_part_end': upload.part_end,
            'on_end': upload.end,
        })
        try:
            remaining = length
            while remaining:
                chunk = stream.read(min(remaining, 256 * 1024))
                if not chunk:
                    raise RequestError('Incomplete upload.')
                parser.write(chunk)
                remaining -= len(chunk)
            parser.finalize()
            if not upload.complete:
                raise RequestError('Incomplete multipart upload.')
            if not isinstance(upload.metadata, dict):
                raise RequestError('Missing manga details.')
            yield upload.metadata, upload.files
        except MultipartParseError as error:
            raise RequestError('Invalid multipart upload.') from error
        finally:
            if upload.file:
                upload.file.close()


class ImportParts:
    def __init__(self, directory):
        self.directory = directory
        self.files = []
        self.file = None
        self.metadata = None
        self.complete = False

    def part_begin(self):
        self.headers = {}
        self.field = bytearray()
        self.value = bytearray()
        self.metadata_bytes = bytearray()
        self.header_bytes = 0

    def header_data(self, buffer, data, start, end):
        self.header_bytes += end - start
        if self.header_bytes > MAX_PART_HEADER_BYTES:
            raise RequestError('Upload field headers are too large.')
        buffer.extend(data[start:end])

    def header_field(self, data, start, end):
        self.header_data(self.field, data, start, end)

    def header_value(self, data, start, end):
        self.header_data(self.value, data, start, end)

    def header_end(self):
        self.headers[bytes(self.field).lower()] = bytes(self.value)
        self.field.clear()
        self.value.clear()

    def headers_finished(self):
        disposition, params = parse_options_header(self.headers.get(b'content-disposition', b''))
        name = params.get(b'name')
        if disposition != b'form-data':
            raise RequestError('Unexpected upload field.')
        if name == b'metadata' and self.metadata is None:
            return
        if name != b'page' or not params.get(b'filename'):
            raise RequestError('Unexpected upload field.')
        if len(self.files) >= MAX_PAGES:
            raise RequestError(f'An import can contain at most {MAX_PAGES} files.')
        path = self.directory / str(len(self.files))
        self.file = path.open('wb')
        self.files.append((params[b'filename'].decode('utf-8', errors='replace'), path))

    def part_data(self, data, start, end):
        if self.file:
            self.file.write(memoryview(data)[start:end])
            return
        if len(self.metadata_bytes) + end - start > MAX_METADATA_BYTES:
            raise RequestError('Manga details are too large.')
        self.metadata_bytes.extend(data[start:end])

    def part_end(self):
        if self.file:
            self.file.close()
            self.file = None
        else:
            self.metadata = json.loads(self.metadata_bytes)
            if not isinstance(self.metadata, dict):
                raise RequestError('Missing manga details.')

    def end(self):
        self.complete = True
