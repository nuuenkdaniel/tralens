#!/bin/sh
set -eu
umask 077

download() {
    name=$1
    revision=$2
    hf download "PaddlePaddle/$name" \
        README.md inference.yml inference.json inference.pdiparams \
        --revision "$revision" --local-dir "/models/$name"
}

download PP-OCRv6_medium_det 8e0f56fb2ef86b461d99cfc7ac5c137738985f61
download PP-OCRv6_medium_rec e5a92bcbc5cc1b494628e458d267778f0704fd7c
PYTHONPATH=/app/backend/src python -c 'from tralens.ocr.paddle_profiles import resolve_profile; resolve_profile(model_root="/models", device="cpu")'
