#!/bin/bash
# Held-out comparison of sites and image quality in NeuroGAMLSS z-maps: terms of the GAMLSS
# (training sites, IQMs in the normative model or in the test controls) versus ComCat on
# the test data or on their normal scores (scripts/harmonization_comparison.py).
#
# Usage: scripts/run_harmonization_comparison.sh [output folder] [further options]
#   e.g. scripts/run_harmonization_comparison.sh harmonization --samples ABIDE,ADNI
#        scripts/run_harmonization_comparison.sh harmonization --features 5000   (faster)
# Environment:
#   NG_DATA     folder with TrainingData and TestData (default ~/Dropbox/neuroGAMLSS-TestData)
#   COMCAT_DIR  ComCat repository (default ~/GitHub/ComCat)
#   JOBS        threads for fitting the normative models (default 8)
#
# The normative models (adult and child, with TIV and with TIV + IQMs) are fitted once and
# saved in the output folder; a second call reuses them.  Results: <out>/comparison.csv
# (all metrics) and <out>/comparison.md (tables), logs in <out>/*.log.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-harmonization}
shift || true
DATA=${NG_DATA:-$HOME/Dropbox/neuroGAMLSS-TestData}
export COMCAT_DIR=${COMCAT_DIR:-$HOME/GitHub/ComCat}
mkdir -p "$OUT"
[ -f "$COMCAT_DIR/comcat.py" ] || { echo "ComCat not found in $COMCAT_DIR (set COMCAT_DIR)"; exit 1; }
[ -d "$DATA/TrainingData" ] && [ -d "$DATA/TestData" ] || {
  echo "$DATA must contain TrainingData and TestData (set NG_DATA)"; exit 1; }

run() {   # part, then the further options
  local part=$1
  shift
  echo "### $part  $(date)"
  python3 "$HERE/harmonization_comparison.py" --root "$DATA" --out "$OUT" --part "$part" "$@" \
    2>&1 | tee "$OUT/$part.log"
}
run models --jobs "${JOBS:-8}" "$@"
run compare "$@"
run report "$@"
echo "Tables: $OUT/comparison.md"
