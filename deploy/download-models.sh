#!/bin/sh
set -eu

destination=${1:-/models}
mkdir -p "$destination"
download() {
    name=$1
    revision=$2
    checksum=$3
    model_dir="$destination/$name"
    hf download "PaddlePaddle/$name" \
        README.md inference.yml inference.json inference.pdiparams \
        --revision "$revision" --local-dir "$model_dir"
    printf '%s  %s\n' "$checksum" "$model_dir/inference.pdiparams" | sha256sum -c -
    rm -rf "$model_dir/.cache"
}

download PP-OCRv6_medium_det 8e0f56fb2ef86b461d99cfc7ac5c137738985f61 85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960
download PP-OCRv6_medium_rec e5a92bcbc5cc1b494628e458d267778f0704fd7c 1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319
download korean_PP-OCRv5_mobile_rec c02ecaf1f22bfd1c618cce154fd19185b47e663a cac3e5f12cf04aaa77f6a5bc704e4e736ef2908476551891d84b41b4e9090462
