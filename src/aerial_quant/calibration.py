"""Select a fixed-budget calibration set using declared coverage strata.

Greedy coverage is a research baseline, not a claimed new algorithm. Tags must
come only from training labels/metadata, never from test outcomes.
"""
from collections import Counter
import hashlib
from typing import Mapping, Sequence


def select_coverage(rows: Sequence[dict], budget: int,
                    minimum_coverage: Mapping[str, int], max_per_group: int,
                    seed: int = 0) -> list[str]:
    if budget <= 0 or budget > len(rows) or max_per_group <= 0:
        raise ValueError('invalid budget or group cap')
    if any(type(v) is not int or v < 0 for v in minimum_coverage.values()):
        raise ValueError('coverage counts must be non-negative integers')
    ids = [r['image_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate image ids')
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('calibration candidates must come from train')
        if not row.get('group_id') or not isinstance(row.get('tags'), list):
            raise ValueError('group_id and explicit tags are required')
    selected, covered, groups = [], Counter(), Counter()
    remaining = list(rows)
    while len(selected) < budget:
        eligible = [r for r in remaining if groups[r['group_id']] < max_per_group]
        if not eligible:
            raise ValueError('group cap prevents reaching budget')
        def priority(row):
            gain = sum(max(0, minimum_coverage[t] - covered[t]) /
                       max(1, minimum_coverage[t])
                       for t in set(row['tags']) if t in minimum_coverage)
            tie = hashlib.sha256(f"{seed}:{row['image_id']}".encode()).hexdigest()
            return (-gain, tie)
        pick = min(eligible, key=priority)
        selected.append(pick['image_id'])
        covered.update(set(pick['tags']))
        groups[pick['group_id']] += 1
        remaining.remove(pick)
    missing = {t: n - covered[t] for t, n in minimum_coverage.items() if covered[t] < n}
    if missing:
        raise ValueError(f'greedy selection did not meet coverage: {missing}; '
                         'this does not prove the constraints are infeasible')
    return selected
