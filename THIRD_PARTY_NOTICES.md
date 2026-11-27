# Third-party notices

Tralens is distributed under [GPL-3.0-or-later](LICENSE). Dependencies and model weights have their own terms.

## Container distribution

A published image carries the Tralens source repository and revision in its OCI labels. The exact corresponding Tralens source, including build instructions and scripts, is available from the revision URL in the image's `org.opencontainers.image.documentation` label.

The image includes Debian packages from the official Python base image and from the Dockerfile. See `/usr/share/doc/<package>/copyright` and `/usr/share/common-licenses` for package terms. `/app/debian-sources.tsv` lists the installed binary packages and exact Debian source names and versions. Use [Debian Snapshot's source archive](https://snapshot.debian.org/package/) to retrieve those versions. The [official Python image source](https://github.com/docker-library/python) includes its build instructions and [CPython](https://github.com/python/cpython) publishes source and license texts for each version.

`uv.lock` lists the versions and hashes of Python dependencies, with source archives where upstream publishes them. The image preserves wheel license files and metadata alongside the installed packages under `/app/.venv/lib`. In particular:

| Component | Primary license and source |
| --- | --- |
| PaddleOCR 3.7.0, PaddlePaddle 3.3.1 and PaddleX 3.7.2 | Apache-2.0. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) · [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) · [PaddleX](https://github.com/PaddlePaddle/PaddleX) |
| AIStudio SDK 0.3.9 | Apache-2.0 per its [release metadata](https://pypi.org/project/aistudio-sdk/0.3.9/). |
| OpenCV contrib wheel 4.10.0.84 | OpenCV uses Apache-2.0. The Linux wheel includes separately licensed native libraries such as LGPL FFmpeg and Qt. See the wheel's `LICENSE-3RD-PARTY.txt`, its [build sources](https://github.com/opencv/opencv-python/tree/4.10.0.84), and [OpenCV sources](https://github.com/opencv/opencv/tree/4.10.0). |
| Pillow 13.0.0 | HPND. Its wheels also include separately licensed image and font libraries. [Pillow sources and notices](https://github.com/python-pillow/Pillow) |
| scikit-learn, NumPy, SciPy | BSD-3-Clause, with additional bundled library notices in the wheels. [scikit-learn](https://github.com/scikit-learn/scikit-learn) · [NumPy](https://github.com/numpy/numpy) · [SciPy](https://github.com/scipy/scipy) |
| ONNX Runtime 1.23.2 | MIT, with bundled component notices. [ONNX Runtime](https://github.com/microsoft/onnxruntime/tree/v1.23.2) |
| pypdfium2 4.30.0 | Apache-2.0 or BSD-3-Clause, with PDFium and third-party notices in the wheel. [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) |
| SentencePiece 0.2.1 and Tokenizers 0.21.4 | Apache-2.0. [SentencePiece](https://github.com/google/sentencepiece/tree/v0.2.1) · [Tokenizers](https://github.com/huggingface/tokenizers/tree/v0.21.4) |
| DejaVu Sans | The container uses Debian's unmodified `fonts-dejavu-core` package. The package includes Bitstream Vera and Arev license texts. [Debian package and source](https://packages.debian.org/bookworm/fonts-dejavu-core) |
| Comic Neue | Shouts and handwriting use Debian’s packaged faces. The package includes the SIL Open Font License 1.1 text. |
| Fugashi, jaconv, and UniDic Lite | The Japanese reader uses the packaged tokenizer, kana converter, and dictionary for word readings. Their license files are included with the installed Python distributions. |

Optional native Android builds pull Capacitor and Android dependencies separately. Their pinned npm versions are in `frontend/android/package-lock.json`. The beta container does not include Android build tools or APKs.

## Model and evaluation assets

The image includes the unmodified pinned [PaddleOCR detector](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det) and [recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec) exports from the upstream Apache-2.0 model repositories. Each model card is packaged alongside its weights under `/opt/tralens/ocr`. The license text is at `/usr/share/common-licenses/Apache-2.0` and the build verifies the weight checksums. Ollama models are installed by the operator. TranslateGemma uses the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Check the terms for every additional OCR, inpainting, or translation model before using or redistributing it.

Experimental font choices, manga OCR snapshots, LaMa, SDXL, and text-mask checkpoints require operator-supplied assets. Redistribution rights for the experimental Comic Text Detector checkpoint have not been established. Local evaluation pages, translated samples, and demo transcripts are excluded from the release tree and image.
