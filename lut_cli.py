#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable
from random import Random

from libs.image_handlers import get_images_from_folder
from libs.lut_api_final import (
    apply_luts_as_data_urls,
    apply_luts_to_directory,
    batch_apply_luts,
    find_luts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply LUTs to images using libs/lut_api_final.py",
    )
    parser.add_argument(
        "input_path",
        help="Image file or folder of images",
    )
    parser.add_argument(
        "lut_source",
        nargs="+",
        help="LUT file(s) or a folder containing LUTs",
    )
    parser.add_argument(
        "--mode",
        choices=("data_url", "save", "batch"),
        default="data_url",
        help="data_url=single image to data URLs; save=single image to directory; batch=folder to directory",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for save/batch modes",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max number of images to process when input is a folder",
    )
    parser.add_argument(
        "--max-luts",
        type=int,
        default=None,
        help="Max number of LUTs to use",
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=None,
        help="Seed for deterministic random selection of images/LUTs",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Max output dimension (width/height) for resizing",
    )
    return parser.parse_args()


def _resolve_luts(lut_source: Iterable[str], limit: int | None, seed: float | None) -> list[Path]:
    sources = [Path(p) for p in lut_source]
    if len(sources) == 1 and sources[0].is_dir():
        return find_luts(sources[0], limit=limit, seed=seed)
    luts = sources
    if seed is not None:
        rng = Random(seed)
        rng.shuffle(luts)
    if limit is not None:
        luts = luts[:limit]
    return luts


def _run_data_url(image_path: Path, luts: list[Path], max_size: int | None) -> None:
    results = apply_luts_as_data_urls(
        image=image_path,
        lut_paths=luts,
        max_size=max_size,
    )
    print(json.dumps(results, indent=2))


def _run_save(image_path: Path, luts: list[Path], output_dir: Path, max_size: int | None) -> None:
    results = apply_luts_to_directory(
        image=image_path,
        lut_paths=luts,
        output_dir=output_dir,
        max_size=max_size,
    )
    print(json.dumps(results, indent=2))


def _run_batch(
    image_folder: Path,
    luts: list[Path],
    output_dir: Path,
    max_images: int | None,
    seed: float | None,
    max_size: int | None,
) -> None:
    images = get_images_from_folder(image_folder, max_images or 10**9, seed=seed)
    if not images:
        raise SystemExit(f"No images found in {image_folder}")
    results = batch_apply_luts(
        image_paths=images,
        lut_paths=luts,
        output_dir=output_dir,
        max_size=max_size,
    )
    print(json.dumps(results, indent=2))


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    luts = _resolve_luts(args.lut_source, limit=args.max_luts, seed=args.seed)
    if not luts:
        raise SystemExit("No LUTs found.")

    if input_path.is_dir():
        if args.mode != "batch":
            raise SystemExit("Folder input requires --mode batch.")
        _run_batch(
            image_folder=input_path,
            luts=luts,
            output_dir=Path(args.output_dir),
            max_images=args.max_images,
            seed=args.seed,
            max_size=args.max_size,
        )
        return

    if args.mode == "data_url":
        _run_data_url(input_path, luts, args.max_size)
    elif args.mode == "save":
        _run_save(input_path, luts, Path(args.output_dir), args.max_size)
    else:
        raise SystemExit("File input requires --mode data_url or --mode save.")


if __name__ == "__main__":
    main()
