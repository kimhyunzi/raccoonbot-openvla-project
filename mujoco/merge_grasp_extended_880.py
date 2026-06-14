from pathlib import Path
import json
import shutil
from datetime import datetime
from collections import defaultdict

SRC_GRASP = Path("raccoon_grasp_colored_cylinder")
SRC_EXTENDED = Path("raccoon_extended_tasks_480")
OUT_ROOT = Path("raccoon_final_multitask_dataset")

COLORS = ("red", "blue", "green", "yellow")

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def infer_color_from_instruction(instruction: str):
    lower = instruction.lower()
    for color in COLORS:
        if color in lower:
            return color
    return ""

def list_episodes(root: Path):
    eps = sorted([p for p in root.glob("episode_*") if p.is_dir()])
    return eps

def copy_episode(src_ep: Path, dst_ep: Path, new_episode_id: int, default_task_type: str):
    if dst_ep.exists():
        shutil.rmtree(dst_ep)
    shutil.copytree(src_ep, dst_ep)

    meta_path = dst_ep / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")

    meta = read_json(meta_path)

    old_episode_id = meta.get("episode_id", None)
    meta["episode_id"] = int(new_episode_id)

    if not meta.get("task_type"):
        meta["task_type"] = default_task_type

    if not meta.get("target_color"):
        meta["target_color"] = infer_color_from_instruction(str(meta.get("instruction", "")))

    meta["source_dataset"] = str(src_ep.parent.name)
    meta["source_episode_dir"] = str(src_ep.name)
    meta["old_episode_id"] = old_episode_id

    write_json(meta_path, meta)

def main():
    if not SRC_GRASP.exists():
        raise FileNotFoundError(f"missing source: {SRC_GRASP}")
    if not SRC_EXTENDED.exists():
        raise FileNotFoundError(f"missing source: {SRC_EXTENDED}")

    grasp_eps = list_episodes(SRC_GRASP)
    extended_eps = list_episodes(SRC_EXTENDED)

    print(f"[SOURCE] {SRC_GRASP}: {len(grasp_eps)} episodes")
    print(f"[SOURCE] {SRC_EXTENDED}: {len(extended_eps)} episodes")

    if len(grasp_eps) != 400:
        print(f"[WARN] grasp episode count is {len(grasp_eps)}, expected 400")
    if len(extended_eps) != 480:
        print(f"[WARN] extended episode count is {len(extended_eps)}, expected 480")

    if OUT_ROOT.exists():
        backup = Path(f"{OUT_ROOT.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"[BACKUP] existing {OUT_ROOT} -> {backup}")
        shutil.move(str(OUT_ROOT), str(backup))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    next_id = 1

    print("[MERGE] copying grasp episodes...")
    for src_ep in grasp_eps:
        dst_ep = OUT_ROOT / f"episode_{next_id:06d}"
        copy_episode(src_ep, dst_ep, next_id, default_task_type="grasp")
        next_id += 1

    print("[MERGE] copying extended task episodes...")
    for src_ep in extended_eps:
        dst_ep = OUT_ROOT / f"episode_{next_id:06d}"
        copy_episode(src_ep, dst_ep, next_id, default_task_type="")
        next_id += 1

    final_eps = list_episodes(OUT_ROOT)

    counts = defaultdict(lambda: defaultdict(int))
    success_count = 0
    failed_count = 0

    for ep in final_eps:
        meta = read_json(ep / "meta.json")
        task = meta.get("task_type", "unknown")
        color = meta.get("target_color", "unknown")
        counts[task][color] += 1
        if bool(meta.get("success", False)):
            success_count += 1
        else:
            failed_count += 1

    print()
    print("[DONE] merged dataset created")
    print(f"output root: {OUT_ROOT}")
    print(f"total episodes: {len(final_eps)}")
    print(f"success episodes: {success_count}")
    print(f"failed episodes: {failed_count}")
    print()
    print("[TASK/COLOR COUNTS]")
    for task in sorted(counts.keys()):
        print(f"  {task}: {dict(sorted(counts[task].items()))}")

    expected_total = len(grasp_eps) + len(extended_eps)
    if len(final_eps) != expected_total:
        raise RuntimeError(f"final count mismatch: {len(final_eps)} != {expected_total}")

    expected_names = [f"episode_{i:06d}" for i in range(1, expected_total + 1)]
    actual_names = [p.name for p in final_eps]
    missing = sorted(set(expected_names) - set(actual_names))
    extra = sorted(set(actual_names) - set(expected_names))

    if missing or extra:
        print("[WARN] sequence mismatch")
        print("missing:", missing[:10])
        print("extra:", extra[:10])
    else:
        print("[OK] episode numbering is continuous")

if __name__ == "__main__":
    main()
