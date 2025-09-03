#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

def lowest_common_parent(a: Path, b: Path) -> Path:
    ap, bp = a.resolve(), b.resolve()
    common = []
    for x, y in zip(ap.parts, bp.parts):
        if x == y:
            common.append(x)
        else:
            break
    return Path(*common) if common else Path.cwd()

def collect_pairs(src_root: Path, tgt_root: Path, strict: bool):
    pairs, missing = [], []
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = p.relative_to(src_root)
            q = tgt_root / rel
            if q.is_file():
                pairs.append(rel)  # store the relative path once; same for src/tgt
            else:
                missing.append(rel)
    if not pairs:
        raise RuntimeError("No matched image pairs found. Check your folders and extensions.")
    if missing and strict:
        examples = "\n".join(str(m) for m in missing[:20])
        more = f"\n... and {len(missing)-20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"{len(missing)} source images have no matching target under {tgt_root}.\n"
            f"Examples:\n{examples}{more}"
        )
    return pairs, missing

def main():
    parser = argparse.ArgumentParser(
        description="Create split.csv pairing images by matching relative paths, "
                    "prefixing with the actual src/tgt basenames (e.g., TrainA/, TrainB/)."
    )
    parser.add_argument("--src", required=True, type=Path, help="Path to source folder (e.g., .../TrainA)")
    parser.add_argument("--tgt", required=True, type=Path, help="Path to target folder (e.g., .../TrainB)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV path (default: <lowest_common_parent>/split.csv)")
    parser.add_argument("--exts", nargs="*", default=None,
                        help="Override image extensions. Example: --exts .png .tif")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any src image lacks a corresponding tgt file.")
    args = parser.parse_args()

    src_root, tgt_root = args.src, args.tgt
    if not src_root.is_dir():
        print(f"[error] --src not a directory: {src_root}", file=sys.stderr); sys.exit(2)
    if not tgt_root.is_dir():
        print(f"[error] --tgt not a directory: {tgt_root}", file=sys.stderr); sys.exit(2)

    # Basenames that must appear in CSV (e.g., TrainA, TrainB)
    src_base = src_root.name
    tgt_base = tgt_root.name

    global IMG_EXTS
    if args.exts:
        IMG_EXTS = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts}

    out_path = args.out
    if out_path is None:
        out_path = lowest_common_parent(src_root, tgt_root) / "split.csv"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs, missing = collect_pairs(src_root, tgt_root, strict=args.strict)

    # Write CSV: "<src_base>/<rel>,<tgt_base>/<rel>"
    lines = []
    for rel in pairs:
        left = (Path(src_base) / rel).as_posix()
        right = (Path(tgt_base) / rel).as_posix()
        lines.append(f"{left},{right}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] Wrote {len(lines)} pairs to {out_path}")
    if missing:
        print(f"[warn] {len(missing)} src files had no matching tgt and were skipped "
              f"(use --strict to fail).")

if __name__ == "__main__":
    main()
