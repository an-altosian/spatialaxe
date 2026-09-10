#!/bin/bash
# Run the dev-HEAD reference image_qc.py with dev-HEAD's OWN xenium_helpers,
# shadowing the container's older baked copy via PYTHONPATH.
set -u

SAMPLE=XETG00378__0061499__R2_control
BUNDLE=/data/bundle
OUT=/tmp/ab/ref
mkdir -p /tmp/mpl /tmp/numba "$OUT"

export PYTHONPATH=/helpers

echo "=================== REFERENCE RUN (dev HEAD cd87a5d) ==================="
echo "image_qc.py    : $(wc -l < /ref/image_qc.py) lines, md5 $(md5sum /ref/image_qc.py | cut -d' ' -f1)"
echo "snr_metrics.py : $(wc -l < /ref/snr_metrics.py) lines, md5 $(md5sum /ref/snr_metrics.py | cut -d' ' -f1)"
python3 -c "
import xenium_helpers.utils as u
print('xenium_helpers resolved to:', u.__file__)
print('has read_xenium_analysis_sw_version:', hasattr(u, 'read_xenium_analysis_sw_version'))
"

cd /ref || exit 1
START=$(date +%s)
python3 image_qc.py --xenium-bundle-dir "$BUNDLE" --outdir "$OUT" --sample-id "$SAMPLE"
echo "REF_EXIT=$?"
echo "REF_SECONDS=$(( $(date +%s) - START ))"
echo "--- output inventory:"
find "$OUT" -type f | sed "s|$OUT/||" | sort
echo "REF_DONE"
