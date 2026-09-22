#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${PROJECT_ROOT}/safe-control-gym"
ENV_PREFIX="${PROJECT_ROOT}/conda311"
SCG_COMMIT="6b5391d014f36fdfa0f9d22d92c77387e5274308"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Install Miniforge or Anaconda first." >&2
  exit 1
fi

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
  git clone https://github.com/learnsyslab/safe-control-gym.git "${SOURCE_ROOT}"
fi

git -C "${SOURCE_ROOT}" checkout "${SCG_COMMIT}"
conda create -y -p "${ENV_PREFIX}" python=3.11 pip
conda install -y -p "${ENV_PREFIX}" -c conda-forge \
  pytorch=2.10 numpy=2.4 scipy scikit-learn matplotlib pybullet cddlib gmp

"${ENV_PREFIX}/bin/pip" install \
  casadi==3.8.1 \
  cvxpy==1.9.3 \
  dict-deep==4.1.2 \
  gpytorch==1.15.2 \
  gymnasium==0.28.1 \
  imageio==2.37.4 \
  Mosek==11.2.4 \
  munch==4.0.0 \
  pytope==0.0.4 \
  PyYAML==6.0.3 \
  scikit-optimize==0.10.2 \
  tensorboard==2.20.0 \
  termcolor==3.3.0

CFLAGS="-I${ENV_PREFIX}/include" \
LDFLAGS="-L${ENV_PREFIX}/lib" \
  "${ENV_PREFIX}/bin/pip" install pycddlib==2.1.8.post1
"${ENV_PREFIX}/bin/pip" install --no-deps -e "${SOURCE_ROOT}"

echo
echo "Setup complete. Run:"
echo "${ENV_PREFIX}/bin/python ${PROJECT_ROOT}/reproduce.py --scg-root ${SOURCE_ROOT} --output ${PROJECT_ROOT}/results"
