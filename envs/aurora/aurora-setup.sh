#! /bin/bash

cd /lus/flare/projects/Diaspora/mike/cascade

module load frameworks
python3 -m venv ./venv --system-site-packages
source ./venv/bin/activate

pip install -e ".[mace]"

# fairchem-core can't be installed normally on Aurora:
#  - it requires torch~=2.13.0; Aurora's XPU torch is 2.13.0a0, a pre-release that pip
#    sorts before 2.13.0, so pip would replace it with a CUDA build from PyPI
#  - it requires e3nn>=0.5; mace-torch pins e3nn==0.4.4 (UMA only uses o3.FromS2Grid/ToS2Grid,
#    which 0.4.4 has)
#  - it requires nvalchemi-toolkit-ops (NVIDIA-only; imported optionally)
# So install fairchem itself without dependencies, then its remaining dependencies, with torch
# and e3nn pinned to what is already installed.
CONSTRAINTS=$(mktemp)
python -c "import torch, e3nn; print(f'torch=={torch.__version__}\ne3nn=={e3nn.__version__}')" > "$CONSTRAINTS"
pip install --no-deps "fairchem-core==2.23.0"
pip install -c "$CONSTRAINTS" \
    "ase-db-backends>=0.10.0" "clusterscope==0.0.18" backoff "hydra-core>=1.3" \
    "monty>=2026.2.18" submitit torchtnt wandb
rm -f "$CONSTRAINTS"
# Note: e3nn 0.4.4 loads its constants with torch.load, which torch>=2.6 refuses by default.
# Importing mace sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, which fixes it; a process that uses
# fairchem without importing mace first must set that variable itself.

conda create -p ./conda-env python=3.12.12
conda activate ./conda-env
conda install -y postgresql
