"""Preview allowlisted legacy .pt removal; --apply requires a successful idle-process check."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.mirror_roi import sha256

ROOT = Path(__file__).resolve().parents[1]
OLD_RUNS = {"baseline_MSE", "baseline_kld_cc", "tsm-attn-bs1-amp", "tsm_attn_pool_kld_cc",
            "tsm-attn-frozen-seed123", "tsm_attn_pool_mse_cc_e1-3", "tsm-attn-aug-seed123",
            "temporal-mean", "tsm-attn-paper-hparams", "tsm-attn-aug-fast-seed123"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    base = (ROOT / "checkpoints").resolve()
    anchors = [base / "legacy/drivergaze360.pt", base / "tsm-attn-aug-fast-seed123/best.pt"]
    if not all(path.is_file() for path in anchors):
        raise FileNotFoundError("Required author/shared initialization checkpoint missing.")
    candidates = []
    for path in sorted(base.rglob("*.pt")):
        rel = path.relative_to(base)
        eligible = (rel.as_posix() == "legacy/on_dada.pt" or rel.parts[0] in OLD_RUNS or
                    rel.parts[0].startswith("dg360_"))
        resolved = path.resolve()
        if eligible and base in resolved.parents and resolved not in anchors:
            candidates.append(resolved)
    print(json.dumps({"mode": "apply" if args.apply else "preview_only",
                      "candidates": [{"path": str(p), "bytes": p.stat().st_size} for p in candidates]}, indent=2))
    if not args.apply:
        return
    # Query failures/permissions fail closed. A non-authoritative empty listing is insufficient.
    command = "$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            text=True, capture_output=True, check=True)
    processes = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(processes, dict):
        processes = [processes]
    processes = [process for process in processes if int(process["ProcessId"]) != os.getpid()]
    if processes:
        raise RuntimeError("Python processes exist; close training/inference before cleanup.")
    manifest = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size}
                for p in candidates]
    out = ROOT / "results/unified_protocol"
    out.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"cleanup_{stamp}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in candidates:
        path.unlink()
    print(f"Removed {len(candidates)} legacy .pt files; logs and reports preserved.")


if __name__ == "__main__":
    main()
