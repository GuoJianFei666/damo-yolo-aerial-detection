"""Audit raw VisDrone labels and evaluation partitions without changing labels.

No official AP is implemented here. Keep ignored annotations and use the official
VisDrone evaluator for benchmark claims. Size strata below are project-specific.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import re
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    width: float
    height: float
    score: int
    category: int
    truncation: int
    occlusion: int

    @property
    def evaluable(self) -> bool:
        return self.score == 1 and 1 <= self.category <= 10


def parse_annotation(text: str) -> list[Box]:
    """Read the eight-field ground-truth format; do not silently repair rows."""
    boxes: list[Box] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().rstrip(',').split(',')
        if len(parts) != 8:
            raise ValueError(f'line {line_number}: expected 8 fields, got {len(parts)}')
        try:
            values = [float(item.strip()) for item in parts]
        except ValueError as exc:
            raise ValueError(f'line {line_number}: invalid numeric field') from exc
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'line {line_number}: non-finite value')
        if any(v != int(v) for v in values[4:]):
            raise ValueError(f'line {line_number}: label flags must be integers')
        left, top, width, height = values[:4]
        score, category, truncation, occlusion = map(int, values[4:])
        if min(left, top) < 0 or width <= 0 or height <= 0:
            raise ValueError(f'line {line_number}: invalid box geometry')
        if score not in (0, 1) or category not in range(12):
            raise ValueError(f'line {line_number}: invalid ground-truth score/category')
        if truncation not in (0, 1, 2) or occlusion not in (0, 1, 2):
            raise ValueError(f'line {line_number}: invalid truncation/occlusion flag')
        boxes.append(Box(left, top, width, height, score, category, truncation, occlusion))
    return boxes


def image_summary(boxes: Iterable[Box], image_width: int, image_height: int,
                  input_size: int = 640, tiny_short_side: float = 16.0) -> dict:
    """Report effective sizes after aspect-ratio preserving resizing.

    Letterbox offset does not change size. This is NOT COCO AP_small. Out-of-image
    boxes are listed as anomalies and are not clipped by this audit function.
    """
    if min(image_width, image_height, input_size) <= 0 or tiny_short_side <= 0:
        raise ValueError('image sizes and tiny threshold must be positive')
    boxes = list(boxes)
    valid = [b for b in boxes if b.evaluable]
    scale = min(input_size / image_width, input_size / image_height)
    anomalies = [i for i, b in enumerate(boxes)
                 if b.left + b.width > image_width or b.top + b.height > image_height]
    tiny = sum(min(b.width, b.height) * scale < tiny_short_side for b in valid)
    return {
        'raw_boxes': len(boxes), 'evaluable_boxes': len(valid),
        'ignored_or_other_boxes': len(boxes) - len(valid),
        'class_counts': dict(sorted(Counter(b.category for b in valid).items())),
        'heavy_occlusion_boxes': sum(b.occlusion == 2 for b in valid),
        'tiny_after_resize_boxes': tiny,
        'tiny_definition': f'min(resized_width,resized_height)<{tiny_short_side:g}px',
        'input_size': input_size, 'resize_scale': scale,
        'out_of_image_row_indices': anomalies,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_splits(partitions: Mapping[str, list[dict]]) -> dict:
    """Validate train/calibration/select/confirm/test manifests.

    Calibration must be a subset of training images. All other partitions must
    be disjoint by id, image hash, and supplied scene/flight group id. This does
    not discover near-duplicates or verify the truth of supplied group metadata.
    """
    names = {'train', 'calibration', 'select', 'confirm', 'test'}
    if set(partitions) != names:
        raise ValueError(f'exactly these partitions are required: {sorted(names)}')
    index = {}
    for name, rows in partitions.items():
        if not rows:
            raise ValueError(f'{name}: empty partition')
        by_id = {}
        seen_hashes = set()
        for row in rows:
            if not all(isinstance(row.get(k), str) and row[k]
                       for k in ('image_id', 'sha256', 'group_id')):
                raise ValueError(f'{name}: image_id/sha256/group_id required')
            if not re.fullmatch(r'[a-f0-9]{64}', row['sha256']):
                raise ValueError(f'{name}: invalid sha256')
            if row['image_id'] in by_id or row['sha256'] in seen_hashes:
                raise ValueError(f'{name}: duplicate id or exact image content')
            by_id[row['image_id']] = row
            seen_hashes.add(row['sha256'])
        index[name] = by_id
    for image_id, row in index['calibration'].items():
        parent = index['train'].get(image_id)
        if parent is None or any(parent[k] != row[k] for k in ('sha256', 'group_id')):
            raise ValueError('calibration must be an exact subset of train')
    ordered = sorted(names)
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            if {left, right} == {'train', 'calibration'}:
                continue
            for key in ('image_id', 'sha256', 'group_id'):
                overlap = {r[key] for r in index[left].values()} & {
                    r[key] for r in index[right].values()}
                if overlap:
                    raise ValueError(f'{left}/{right}: overlap in {key}')
    return {'status': 'passed', 'counts': {k: len(v) for k, v in partitions.items()},
            'checks': ['image_id', 'exact_image_hash', 'supplied_group_id'],
            'not_checked': ['near_duplicates', 'correctness_of_group_metadata']}
