from pathlib import Path
from random import Random
from typing import Iterable, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

def find_images_recursive(folder: str | Path) -> List[Path]:
    """Return all image file paths under folder (recursive)."""
    root = Path(folder).expanduser().resolve()
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def pick_shuffled(paths: Iterable[Path], k: int, seed: int | float | None = None) -> List[Path]:
    """Shuffle deterministically with seed and pick top-k."""
    paths = list(paths)
    rng = Random(seed)
    rng.shuffle(paths)
    return paths[:k]

def get_images_from_folder(folder: str | Path, max_count: int, seed: int | float | None = None) -> List[Path]:
    """Runner: gather images recursively, shuffle with seed, and return up to max_count paths."""
    images = find_images_recursive(folder)
    return pick_shuffled(images, max_count, seed)
