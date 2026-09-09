# Container for the image QC module (analysis + Quarto report).
# Built from environment.yml in this directory. Tagged and pushed as
# altoslabscom/image_qc:2.0.0 (to be migrated to the nf-core org for release).
#
#   docker build -t altoslabscom/image_qc:2.0.0 modules/local/image_qc
#
# Note: behind a corporate proxy / private conda mirror you may need to provide a
# CA bundle and channel config at build time (e.g. via BuildKit --mount=type=secret);
# a standard build resolves the pinned packages directly from conda-forge/bioconda.
FROM mambaorg/micromamba:2.0.5
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean -a -f -y \
    && rm -f /tmp/environment.yml
# spatialqc is installed from a wheel built out of the standalone repository
# (github.com/altos-labs/spatialqc), NOT from PyPI: the package is deliberately
# unpublished, so pinning a PyPI version here would make the image unbuildable.
# Build the wheel into this directory before building the image:
#
#   python -m build --wheel --outdir modules/local/<module> packages/spatialqc
#
# --no-deps is deliberate. Every dependency of the extras this module needs is
# already pinned by conda in environment.yml above, so letting pip resolve them
# would re-solve the scientific stack and can silently shadow the conda builds.
COPY --chown=$MAMBA_USER:$MAMBA_USER spatialqc-*.whl /tmp/
RUN /opt/conda/bin/pip install --no-deps --no-cache-dir /tmp/spatialqc-*.whl \
    && rm -f /tmp/spatialqc-*.whl
ENV PATH="/opt/conda/bin:$PATH" \
    QUARTO_DENO=/opt/conda/bin/deno \
    QUARTO_DENO_DOM=/opt/conda/lib/deno_dom.so \
    QUARTO_PANDOC=/opt/conda/bin/pandoc \
    QUARTO_ESBUILD=/opt/conda/bin/esbuild \
    QUARTO_TYPST=/opt/conda/bin/typst \
    QUARTO_DART_SASS=/opt/conda/bin/sass \
    QUARTO_SHARE_PATH=/opt/conda/share/quarto \
    QUARTO_CONDA_PREFIX=/opt/conda \
    ES2_LIBRARY=/opt/conda/lib/libGLESv2.so.2
