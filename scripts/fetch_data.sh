#!/bin/sh
# banking77 (PolyAI, CC BY 4.0) and MASSIVE 1.1 en-US (Amazon, CC BY 4.0), from the original sources, checked by SHA-256.
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/raw/massive
for s in train test; do
  curl -fsSL -o "data/raw/$s.csv" "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/$s.csv"
done
tmp=$(mktemp -d)
curl -fsSL -o "$tmp/massive.tar.gz" https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
shasum -a 256 -c <<SUMS
b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b  data/raw/train.csv
d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d  data/raw/test.csv
4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577  $tmp/massive.tar.gz
SUMS
tar -xzf "$tmp/massive.tar.gz" -C "$tmp"
cp "$(find "$tmp" -name en-US.jsonl | head -1)" data/raw/massive/en-US.jsonl
shasum -a 256 -c <<SUMS
c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e  data/raw/massive/en-US.jsonl
SUMS
rm -rf "$tmp"
PYTHONPATH=src python -c "from ftquant import data, tasks; data.build(); tasks.write_training_files(tasks.get('massive'))"
