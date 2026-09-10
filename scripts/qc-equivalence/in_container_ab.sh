#!/bin/bash
# Run the dev-HEAD reference and the spatialqc port back-to-back in ONE container,
# so the two runs cannot differ by environment. Outputs land under /tmp/ab.
set -u

SAMPLE=XETG00378__0061499__R2_control
BUNDLE=/data/bundle
OUT=/tmp/ab
mkdir -p /tmp/mpl /tmp/numba "$OUT"

echo "=================== ENVIRONMENT ==================="
python3 -c "
import numpy, scipy, sklearn, pandas, tifffile, numba
print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| sklearn', sklearn.__version__)
print('pandas', pandas.__version__, '| tifffile', tifffile.__version__, '| numba', numba.__version__)
"
echo "reference image_qc.py : $(wc -l < /ref/image_qc.py) lines, md5 $(md5sum /ref/image_qc.py | cut -d' ' -f1)"
echo "reference snr_metrics : $(wc -l < /ref/snr_metrics.py) lines, md5 $(md5sum /ref/snr_metrics.py | cut -d' ' -f1)"

echo "=================== RUN A: REFERENCE (dev HEAD) ==================="
cd /ref || exit 1
START=$(date +%s)
python3 image_qc.py --xenium-bundle-dir "$BUNDLE" --outdir "$OUT/ref" --sample-id "$SAMPLE"
echo "A_EXIT=$?"
echo "A_SECONDS=$(( $(date +%s) - START ))"

echo "=================== RUN B: SPATIALQC PORT ==================="
# PYTHONPATH must win over the baked 0.1.0 wheel in site-packages.
export PYTHONPATH=/port_src
cd /tmp || exit 1
python3 -c "import spatialqc, spatialqc.image.qc as m; print('spatialqc resolved to:', spatialqc.__file__); print('qc module:', m.__file__)"
START=$(date +%s)
python3 -m spatialqc.image.qc --xenium-bundle-dir "$BUNDLE" --outdir "$OUT/port" --sample-id "$SAMPLE"
echo "B_EXIT=$?"
echo "B_SECONDS=$(( $(date +%s) - START ))"

echo "=================== OUTPUT INVENTORY ==================="
echo "--- ref:";  find "$OUT/ref"  -type f | sed "s|$OUT/ref/||"  | sort
echo "--- port:"; find "$OUT/port" -type f | sed "s|$OUT/port/||" | sort
echo "AB_DONE"
