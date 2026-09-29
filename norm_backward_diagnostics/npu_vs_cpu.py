"""Run the native NPU backward interface, without importing TENPU or ATK."""
from norm_common import main

if __name__ == "__main__":
    raise SystemExit(main("npu"))
