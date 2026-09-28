#!/bin/sh
# llama.cpp's HF-to-GGUF converter and gguf-py, pinned to the commit that matches llama.cpp 0.5.0 (build 11146).
set -eu
cd "$(dirname "$0")/.."
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp vendor/llama.cpp
cd vendor/llama.cpp
git sparse-checkout set conversion gguf-py
git checkout 7fe450e19305b828c199d602c23a8337aaa1f03b
