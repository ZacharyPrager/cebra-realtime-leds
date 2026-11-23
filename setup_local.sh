#!/bin/bash
# One-time laptop setup:  bash setup_local.sh
# Makes .venv with CPU-only PyTorch + CEBRA + pyserial, records versions, downloads the data.
# If CEBRA/torch won't install on your Python version, use 3.11:  PY=python3.11 bash setup_local.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PY=${PY:-python3}
mkdir -p data env results

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # laptop = CPU only
pip install -r requirements.txt

pip freeze > env/pip_freeze.txt
python - <<'EOF' | tee env/versions.txt
import sys, torch, cebra, numpy, sklearn, serial
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cebra", cebra.__version__)
print("numpy", numpy.__version__, "sklearn", sklearn.__version__, "pyserial", serial.__version__)
EOF

CEBRA_DATADIR="$PWD/data" python -c "
import cebra.datasets
ds = cebra.datasets.init('rat-hippocampus-single-achilles')
print('data ok:', tuple(ds.neural.shape), tuple(ds.continuous_index.shape))"
(cd models && sha256sum -c MANIFEST.txt)

echo "Setup complete. Run:  ./run.sh --preview --steps 400"
