#!/usr/bin/env bash
# Download the authors' raw results from the public bucket gs://hml-public/dd (links in their README)
# into orig_data/<folder>/{Mlist,ks}  (or $DD_ORIG_DIR/<folder>/). Uses gsutil if installed, else curl.
#
#   bash scripts/download_orig_data.sh                     # all folders our figures use
#   bash scripts/download_orig_data.sh cifar10-resnet18k-50k-adam
#   bash scripts/download_orig_data.sh --list              # folders in the bucket
#   bash scripts/download_orig_data.sh --list <folder>     # objects inside one folder
#
# Then:  python -m analysis.orig_data check       (what is available for which figure)
#        python -m analysis.orig_data inspect <folder>   (structure of a CNN folder, see analysis/orig_data.py)
set -u
cd "$(dirname "$0")/.." || exit 1
DEST="${DD_ORIG_DIR:-orig_data}"
API="https://storage.googleapis.com/storage/v1/b/hml-public/o"
HTTP="https://storage.googleapis.com/hml-public/dd"

if [ "${1:-}" = "--list" ]; then
  PREFIX="dd/${2:+$2/}"
  if command -v gsutil >/dev/null; then
    gsutil ls "gs://hml-public/$PREFIX"
  else
    curl -fsS "$API?prefix=$PREFIX&delimiter=/" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for p in d.get("prefixes", []): print(p)
for o in d.get("items", []): print(f"{o[\"name\"]}  ({int(o[\"size\"]) / 1e6:.1f} MB)")'
  fi
  exit $?
fi

FOLDERS=("$@")
if [ ${#FOLDERS[@]} -eq 0 ]; then
  FOLDERS=(cifar10-resnet18k-p15-adam-reps cifar10-resnet18k-50k-adam
           cifar10-mcnn-noaug-sgd cifar10-mcnn-p10-sgd cifar10-mcnn-p20-sgd
           cifar10-mcnn-p20-sgd-rep2 cifar10-mcnn-p20-sgd-rep3 cifar10-mcnn-p20-sgd-rep4 cifar10-mcnn-p20-sgd-rep5
           pct-cifar100-mcnn-p0-sgd-noaug-reps)
fi

FAILED=0
for F in "${FOLDERS[@]}"; do
  mkdir -p "$DEST/$F"
  for OBJ in Mlist ks; do
    OUT="$DEST/$F/$OBJ"
    [ -s "$OUT" ] && { echo "have     $F/$OBJ"; continue; }
    if command -v gsutil >/dev/null; then
      gsutil -q cp "gs://hml-public/dd/$F/$OBJ" "$OUT"
    else
      curl -fsSL -o "$OUT" "$HTTP/$F/$OBJ"
    fi
    if [ -s "$OUT" ]; then
      echo "got      $F/$OBJ ($(du -h "$OUT" | cut -f1))"
    else
      echo "FAILED   $F/$OBJ  - check the object names with: bash scripts/download_orig_data.sh --list $F"
      rm -f "$OUT"; FAILED=1
    fi
  done
done
[ "$FAILED" -eq 0 ] && echo "done -> python -m analysis.orig_data check" \
  || echo "some downloads failed (bucket layout changed or not public any more?) - digitise those figures instead"
