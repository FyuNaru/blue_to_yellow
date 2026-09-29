"""Run NVIDIA TE 2.17 backward primitives, without importing ATK or torch_npu."""
from norm_common import main

if __name__ == "__main__":
    raise SystemExit(main("gpu"))
