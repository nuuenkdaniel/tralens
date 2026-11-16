# Third-party notices

Tralens is licensed under the GNU General Public License v3.0 or later (see
[LICENSE](LICENSE)). It depends on, or can optionally use, the third-party software
and models below. Each keeps its own license; nothing listed here is relicensed
under the GPL. Versions are the ones in the development environment at the time of
writing.

## Python packages

Installed by `pyproject.toml` (directly or as dependencies):

| Package | Version | License |
| --- | --- | --- |
| PaddleOCR (`paddleocr`) | 3.4.0 | Apache-2.0 |
| PaddlePaddle (`paddlepaddle-gpu`) | 3.4.0 | Apache-2.0 |
| PaddleX (`paddlex`) | 3.4.2 | Apache-2.0 |
| Ollama Python client (`ollama`) | 0.6.1 | MIT |
| scikit-learn | 1.7.2 | BSD-3-Clause |
| NumPy | 2.2.6 | BSD-3-Clause |
| Pillow | 10.4.0 | HPND (MIT-CMU) |
| OpenCV (`opencv-contrib-python`) | 4.10.0.84 | Apache-2.0 |
| PyYAML | 6.0.2 | MIT |

Used only by optional features (Manga OCR rereads, Hayai OCR adapter):

| Package | Version | License |
| --- | --- | --- |
| Manga OCR (`manga-ocr`) | 0.1.16 | Apache-2.0 |
| PyTorch (`torch`) | 2.14.1 | BSD-3-Clause, with bundled components under Apache-2.0, MIT and others |
| Hugging Face Transformers | 4.57.6 | Apache-2.0 |

All of these licenses are compatible with GPLv3.

## Models you supply yourself

Tralens does not include or download any model weights. You install and run these
yourself, and their own terms apply to you, including any restrictions on
commercial use or redistribution:

- **PaddleOCR model exports** (detection/recognition): Apache-2.0.
- **`translategemma:27b`** via Ollama: Google's Gemma Terms of Use.
- **`qwen3.8`** vision model via Ollama: the Qwen model license for that release.
- **Manga OCR model snapshot**: check the license on the model's Hugging Face page.
- **`comictextdetector.pt.onnx`** (optional `--text-mask-model`): deployment and
  redistribution rights have not been confirmed. Treat it as local evaluation only.

## Fonts

Tralens ships no fonts. Rendering uses only fonts you list in your own font
manifest, each with its own license file. Font licenses are your responsibility.

## Test data

Some local evaluation pages come from *Ubunchu!* by Hiroshi Seo (published by ASCII
MEDIA WORKS Inc.), released under Creative Commons Attribution-NonCommercial 3.0
(<https://creativecommons.org/licenses/by-nc/3.0/>). These pages are not included in
this repository. Experiment notes in `docs/` refer to them by page name only.
