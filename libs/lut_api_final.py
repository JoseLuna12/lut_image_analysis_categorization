"""
# 1️⃣ Single image → data URLs (for web APIs)
results = apply_luts_as_data_urls(
    image="photo.jpg",
    lut_paths=["lut1.cube", "lut2.cube"],
    max_size=1920
)

# 2️⃣ Single image → save to disk
results = apply_luts_to_directory(
    image="photo.jpg",
    lut_paths=["lut1.cube", "lut2.cube"],
    output_dir="output",
    max_size=1920
)

# 3️⃣ Multiple images × multiple LUTs → parallel processing
results = batch_apply_luts(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    lut_paths=["lut1.cube", "lut2.cube"],
    output_dir="output",
    max_size=1920,
    max_workers=4  # processes in parallel
)

# Find up to 5 random LUTs in a folder (seeded for reproducibility)
luts = find_luts("luts/", limit=5, seed=1234)

# Process 100 photos with 5 LUTs = 500 outputs
photos = list(Path("photos/").glob("*.jpg"))
results = batch_apply_luts(photos, luts, "graded/", max_workers=8)

# Output:
# [10/500] img001.jpg + cinematic.cube
# [20/500] img003.jpg + vintage.cube
# ...
# [500/500] img100.jpg + moody.cube
"""

import base64
import uuid
from random import Random
from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def parse_cube_file(cube_path: Path) -> tuple[int, np.ndarray, float, float]:
    """Parse a .cube LUT file and return (size, 3D_table, domain_min, domain_max)."""
    with open(cube_path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    size, domain_min, domain_max = None, 0.0, 1.0
    lut_data = []

    for line in lines:
        upper = line.upper()
        if upper.startswith("LUT_3D_SIZE"):
            size = int(line.split()[1])
        elif upper.startswith("DOMAIN_MIN"):
            domain_min = float(line.split()[1])
        elif upper.startswith("DOMAIN_MAX"):
            domain_max = float(line.split()[1])
        elif not upper.startswith(("TITLE", "LUT_")):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    lut_data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    pass

    if size is None or len(lut_data) != size**3:
        raise ValueError(f"Invalid cube file: {cube_path}")

    # Reshape to 3D grid and transpose to match RGB indexing
    table = np.array(lut_data, dtype=np.float32).reshape(size, size, size, 3)
    table = np.transpose(table, (2, 1, 0, 3))
    return size, table, domain_min, domain_max


def _resize_if_needed(image: Image.Image, max_size: int | None) -> Image.Image:
    """Resize image if it exceeds max_size in either dimension."""
    if max_size is None:
        return image
    
    w, h = image.size
    if w <= max_size and h <= max_size:
        return image
    
    # Calculate new size maintaining aspect ratio
    if w > h:
        new_w, new_h = max_size, int(h * max_size / w)
    else:
        new_w, new_h = int(w * max_size / h), max_size
    
    return image.resize((new_w, new_h), Image.LANCZOS)


def _resolve_lut_paths(
    lut_paths: Sequence[str | Path] | str | Path,
    limit: int | None = None,
    seed: int | float | None = None,
    extensions: Sequence[str] | None = None,
) -> list[Path]:
    """Normalize LUT inputs to a list of Path objects."""
    if isinstance(lut_paths, (str, Path)):
        candidate = Path(lut_paths)
        if candidate.is_dir():
            return find_luts(candidate, limit=limit, extensions=extensions, seed=seed)
        return [candidate]
    
    lut_list = [Path(p) for p in lut_paths]
    if seed is not None:
        rng = Random(seed)
        rng.shuffle(lut_list)
    if limit is not None:
        lut_list = lut_list[:limit]
    return lut_list


def apply_lut(
    image: Image.Image, 
    lut_path: Path, 
    lut_cache: dict | None = None,
    max_size: int | None = None,
) -> Image.Image:
    """Apply a 3D LUT to an image using trilinear interpolation."""
    # Resize if needed
    image = _resize_if_needed(image, max_size)
    
    # Load or retrieve cached LUT
    cache_key = str(lut_path.resolve())
    if lut_cache is not None and cache_key in lut_cache:
        size, table, domain_min, domain_max = lut_cache[cache_key]
    else:
        size, table, domain_min, domain_max = parse_cube_file(lut_path)
        if lut_cache is not None:
            lut_cache[cache_key] = (size, table, domain_min, domain_max)

    # Convert image to normalized float array
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    # Map to LUT domain and scale to grid coordinates
    arr = (arr - domain_min) / (domain_max - domain_min)
    arr = np.clip(arr, 0, 1) * (size - 1)

    # Get integer indices and fractional parts
    idx = np.floor(arr).astype(np.int32)
    frac = arr - idx

    # Clip indices to valid range
    i0 = np.clip(idx[..., 0], 0, size - 1)
    j0 = np.clip(idx[..., 1], 0, size - 1)
    k0 = np.clip(idx[..., 2], 0, size - 1)
    i1 = np.clip(i0 + 1, 0, size - 1)
    j1 = np.clip(j0 + 1, 0, size - 1)
    k1 = np.clip(k0 + 1, 0, size - 1)

    # Trilinear interpolation: sample 8 corners
    c000 = table[i0, j0, k0]
    c001 = table[i0, j0, k1]
    c010 = table[i0, j1, k0]
    c011 = table[i0, j1, k1]
    c100 = table[i1, j0, k0]
    c101 = table[i1, j0, k1]
    c110 = table[i1, j1, k0]
    c111 = table[i1, j1, k1]

    # Interpolate along each axis
    dx, dy, dz = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]
    c00 = c000 * (1 - dz) + c001 * dz
    c01 = c010 * (1 - dz) + c011 * dz
    c10 = c100 * (1 - dz) + c101 * dz
    c11 = c110 * (1 - dz) + c111 * dz
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    result = c0 * (1 - dx) + c1 * dx

    # Convert back to uint8
    out = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def apply_luts_as_data_urls(
    image: Image.Image | str | Path,
    lut_paths: Sequence[str | Path] | str | Path,
    max_size: int | None = None,
    lut_limit: int | None = None,
    lut_seed: int | float | None = None,
    lut_extensions: Sequence[str] | None = None,
) -> list[dict]:
    """Apply LUTs and return results as base64 data URLs.
    
    Args:
        image: PIL Image or path to image file
        lut_paths: List of paths to .cube LUT files or a directory containing them
        max_size: Maximum dimension (width or height) in pixels. None = no resize.
                  Example: max_size=1920 ensures no dimension exceeds 1920px
        lut_limit: Optional cap on how many LUTs to use (applied after shuffling)
        lut_seed: Seed used when shuffling LUT order (for reproducible random picks)
        lut_extensions: Allowed LUT extensions (defaults to [".cube"])
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    
    lut_cache = {}
    results = []
    lut_paths_list = _resolve_lut_paths(
        lut_paths, limit=lut_limit, seed=lut_seed, extensions=lut_extensions
    )
    
    for lut_path in lut_paths_list:
        transformed = apply_lut(image, lut_path, lut_cache, max_size)
        
        # Encode as data URL
        buffer = BytesIO()
        transformed.save(buffer, format="PNG")
        data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
        results.append({
            "lut": lut_path.name,
            "data_url": data_url,
        })
    
    return results


def apply_luts_to_directory(
    image: Image.Image | str | Path,
    lut_paths: Sequence[str | Path] | str | Path,
    output_dir: str | Path = "output",
    max_size: int | None = None,
    lut_limit: int | None = None,
    lut_seed: int | float | None = None,
    lut_extensions: Sequence[str] | None = None,
) -> list[dict]:
    """Apply LUTs and save results to a directory.
    
    Args:
        image: PIL Image or path to image file
        lut_paths: List of paths to .cube LUT files or a directory containing them
        output_dir: Directory to save output images
        max_size: Maximum dimension (width or height) in pixels. None = no resize.
                  Example: max_size=1920 ensures no dimension exceeds 1920px
        lut_limit: Optional cap on how many LUTs to use (applied after shuffling)
        lut_seed: Seed used when shuffling LUT order (for reproducible random picks)
        lut_extensions: Allowed LUT extensions (defaults to [".cube"])
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lut_cache = {}
    results = []
    lut_paths_list = _resolve_lut_paths(
        lut_paths, limit=lut_limit, seed=lut_seed, extensions=lut_extensions
    )
    
    for lut_path in lut_paths_list:
        transformed = apply_lut(image, lut_path, lut_cache, max_size)
        
        # Save with unique ID
        uid = uuid.uuid4().hex[:8]
        filename = f"{lut_path.stem}_{uid}.jpg"
        output_path = output_dir / filename
        transformed.save(output_path, quality=95)
        
        results.append({
            "lut": lut_path.name,
            "path": str(output_path),
        })
    
    return results


def batch_apply_luts(
    image_paths: Sequence[str | Path],
    lut_paths: Sequence[str | Path] | str | Path,
    output_dir: str | Path = "output",
    max_workers: int = 4,
    max_size: int | None = None,
    lut_limit: int | None = None,
    lut_seed: int | float | None = None,
    lut_extensions: Sequence[str] | None = None,
) -> list[dict]:
    """Apply all LUTs to all images in parallel. Returns saved file info.
    
    Args:
        image_paths: List of paths to image files
        lut_paths: List of paths to .cube LUT files or a directory containing them
        output_dir: Directory to save output images
        max_workers: Number of parallel processes
        max_size: Maximum dimension (width or height) in pixels. None = no resize.
                  Example: max_size=1920 processes much faster for large images
        lut_limit: Optional cap on how many LUTs to use (applied after shuffling)
        lut_seed: Seed used when shuffling LUT order (for reproducible random picks)
        lut_extensions: Allowed LUT extensions (defaults to [".cube"])
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-load all LUTs once
    lut_cache = {}
    lut_paths_list = _resolve_lut_paths(
        lut_paths, limit=lut_limit, seed=lut_seed, extensions=lut_extensions
    )
    for lut_path in lut_paths_list:
        cache_key = str(lut_path.resolve())
        lut_cache[cache_key] = parse_cube_file(lut_path)
    
    # Create all tasks
    tasks = [(img, lut) for img in image_paths for lut in lut_paths_list]
    total = len(tasks)
    
    def process_one(image_path, lut_path):
        """Process a single image+LUT combo."""
        image_path, lut_path = Path(image_path), Path(lut_path)
        img = Image.open(image_path).convert("RGB")
        transformed = apply_lut(img, lut_path, lut_cache, max_size)
        
        uid = uuid.uuid4().hex[:8]
        filename = f"{image_path.stem}_{lut_path.stem}_{uid}.jpg"
        output_path = output_dir / filename
        transformed.save(output_path, quality=95)
        
        return {
            "image": image_path.name,
            "lut": lut_path.name,
            "path": str(output_path),
        }
    
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, img, lut): (img, lut) for img, lut in tasks}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"[{completed}/{total}] {result['image']} + {result['lut']}")
            except Exception as e:
                img, lut = futures[future]
                print(f"[ERROR] {Path(img).name} + {Path(lut).name}: {e}")
    
    return results


def find_luts(
    directory: str | Path,
    limit: int | None = None,
    extensions: Sequence[str] | None = None,
    seed: int | float | None = None,
) -> list[Path]:
    """Find LUT files in a directory, optionally shuffling and limiting the results.
    
    Args:
        directory: Directory to search for LUT files.
        limit: Maximum number of LUTs to return (None = no limit).
        extensions: Allowed LUT extensions (defaults to [".cube"]).
        seed: Seed used to shuffle the list before applying `limit`.
    """
    root = Path(directory).expanduser()
    ext_set = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (extensions or [".cube"])
    }
    lut_files = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in ext_set]
    lut_files.sort(key=lambda p: p.name.lower())
    
    if seed is not None:
        rng = Random(seed)
        rng.shuffle(lut_files)
    
    if limit is not None:
        lut_files = lut_files[:limit]
    
    return lut_files
