"""Smoke-test that core dependencies import correctly."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

checks = []

def check(label, fn):
    try:
        fn()
        checks.append((label, True, ""))
    except Exception as e:
        checks.append((label, False, str(e)))


def main():
    check("torch + cuda", lambda: (
        __import__("torch").cuda.is_available() or (_ for _ in ()).throw(RuntimeError("CUDA not available"))
    ))
    check("decord", lambda: __import__("decord"))
    check("timm", lambda: __import__("timm"))
    check("causal_conv1d", lambda: __import__("causal_conv1d"))
    check("mamba_ssm", lambda: __import__("mamba_ssm"))
    check("mamba_imports shim", lambda: __import__("src.models.utils.mamba_imports"))
    check("src.models.vision_transformer", lambda: __import__("src.models.vision_transformer"))
    check("src.models.videomamba", lambda: __import__("src.models.videomamba"))
    check("app.scaffold", lambda: __import__("app.scaffold"))
    check("evals.scaffold", lambda: __import__("evals.scaffold"))

    print("Dependency verification")
    print("-" * 60)
    failed = 0
    for label, ok, err in checks:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {label}")
        if err:
            print(f"       {err}")
            failed += 1

    print("-" * 60)
    if failed:
        print(f"{failed} check(s) failed.")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
