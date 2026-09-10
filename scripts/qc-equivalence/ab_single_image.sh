#!/bin/bash
# Controlled A/B in ONE image, take 2.
#
# Take 1 failed arm A with "OSError: Read-only file system: './analysis'":
# the reference unpacks the bundle's analysis.tar.gz into its *current working
# directory*, so a read-only cwd kills it. That is a property of the harness,
# not of the code. Both scripts unpack into cwd (image_qc.py:13706,
# qc.py:13041), so both need a writable one. Each arm gets its own here so neither is
# advantaged and neither can clobber the other.
set -u
BUNDLE=/data/bundle
OUT=/tmp/ab
mkdir -p /tmp/mpl /tmp/numba "$OUT" /tmp/refwork /tmp/portwork
cp /refsrc/image_qc.py /refsrc/snr_metrics.py /tmp/refwork/

export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMBA_NUM_THREADS=16
export MPLCONFIGDIR=/tmp/mpl NUMBA_CACHE_DIR=/tmp/numba

echo "=== PROVENANCE ==="
cat /etc/spatialqc-equiv-provenance.txt
python3 - <<'PY'
import xenium_helpers.utils as u, spatialqc
print("xenium_helpers :", u.__file__)
print("  has read_xenium_analysis_sw_version:", hasattr(u,"read_xenium_analysis_sw_version"))
print("spatialqc      :", spatialqc.__version__)
PY
echo "reference md5  : $(md5sum /tmp/refwork/image_qc.py | cut -d' ' -f1) (expect 5012299bc63acc0b93a3367b3ae6fc1f)"

echo "=== ARM A: reference dev HEAD cd87a5d ==="
cd /tmp/refwork || exit 1
A0=$(date +%s)
python3 image_qc.py --xenium-bundle-dir "$BUNDLE" --outdir "$OUT/ref" --sample-id tiny_ileum
A=$?
echo "A_EXIT=$A A_SECONDS=$(( $(date +%s) - A0 ))"

echo "=== ARM B: spatialqc port 882ed8a ==="
cd /tmp/portwork || exit 1
B0=$(date +%s)
spatialqc-image-qc --xenium-bundle-dir "$BUNDLE" --outdir "$OUT/port" --sample-id tiny_ileum
B=$?
echo "B_EXIT=$B B_SECONDS=$(( $(date +%s) - B0 ))"
echo "SUMMARY A_EXIT=$A B_EXIT=$B"
echo "ref files : $(find $OUT/ref -type f 2>/dev/null | wc -l)"
echo "port files: $(find $OUT/port -type f 2>/dev/null | wc -l)"
echo "AB_DONE"
