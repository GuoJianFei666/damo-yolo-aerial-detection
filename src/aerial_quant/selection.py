"""Select from measured complete-engine candidates under declared constraints.

No latency is predicted from FLOPs or layer timings. The solver selects the
fastest feasible record among supplied records, not a global network optimum.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class Constraints:
    max_ap_drop_pp: float = 1.0
    max_small_ap_drop_pp: float = 1.0
    max_group_recall_drop_pp: float = 2.0
    min_latency_reduction: float = 0.15

    def __post_init__(self):
        vals = (self.max_ap_drop_pp, self.max_small_ap_drop_pp,
                self.max_group_recall_drop_pp, self.min_latency_reduction)
        if not all(math.isfinite(v) and v >= 0 for v in vals):
            raise ValueError('constraints must be finite and non-negative')
        if self.min_latency_reduction >= 1:
            raise ValueError('latency reduction must be <1')


def quantile(values: Sequence[float], q: float) -> float:
    if not values or not 0 <= q <= 1 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('positive finite samples and a valid quantile are required')
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] * (high - index) + values[high] * (index - low) if low != high else values[low]


def _validate(row: dict, allow_synthetic: bool, minimum_samples: int):
    if row.get('data_origin') not in ('measured_hardware', 'synthetic_unit_test'):
        raise ValueError('data_origin must be declared')
    if row['data_origin'] == 'synthetic_unit_test' and not allow_synthetic:
        raise ValueError('synthetic records cannot be selected for a real experiment')
    if row.get('partition') != 'select':
        raise ValueError('only select partition records may drive policy selection')
    for key in ('candidate_id', 'protocol_id', 'hardware_id', 'checkpoint_sha256',
                'engine_sha256', 'evaluation_manifest_sha256'):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f'{key} is required')
    if row.get('timing_scope') != 'decode_excluded_e2e':
        raise ValueError('expected preprocessing+H2D+inference+D2H+postprocessing timing')
    for key in ('ap', 'ap_small'):
        if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or not 0 <= row[key] <= 100:
            raise ValueError(f'{key} must be finite percentage points [0,100]')
    groups = row.get('group_recall')
    if not isinstance(groups, dict) or not groups:
        raise ValueError('group_recall must not be empty')
    for key, value in groups.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f'invalid group recall {key}')
    samples = row.get('latency_ms')
    if not isinstance(samples, list) or len(samples) < minimum_samples:
        raise ValueError('insufficient latency samples')
    quantile(samples, 0.95)


def select_measured(records: list[dict], baseline_id: str, constraints: Constraints,
                    *, allow_synthetic: bool = False, minimum_samples: int = 1000) -> dict:
    if minimum_samples < 1:
        raise ValueError('minimum_samples must be positive')
    ids = [r.get('candidate_id') for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate candidate ids')
    for row in records:
        _validate(row, allow_synthetic, minimum_samples)
    baselines = [r for r in records if r['candidate_id'] == baseline_id]
    if len(baselines) != 1:
        raise ValueError('exactly one baseline is required')
    base = baselines[0]
    base_p95 = quantile(base['latency_ms'], 0.95)
    evaluations, feasible = [], []
    for row in records:
        for key in ('data_origin', 'protocol_id', 'hardware_id', 'checkpoint_sha256',
                    'evaluation_manifest_sha256', 'timing_scope'):
            if row[key] != base[key]:
                raise ValueError(f'incomparable records: {key}')
        if set(row['group_recall']) != set(base['group_recall']):
            raise ValueError('group definitions differ')
        if row is base:
            continue
        p95 = quantile(row['latency_ms'], 0.95)
        reasons = []
        if base['ap'] - row['ap'] > constraints.max_ap_drop_pp:
            reasons.append('overall_ap')
        if base['ap_small'] - row['ap_small'] > constraints.max_small_ap_drop_pp:
            reasons.append('small_ap')
        if max(base['group_recall'][g] - row['group_recall'][g]
               for g in base['group_recall']) > constraints.max_group_recall_drop_pp:
            reasons.append('group_recall')
        if 1 - p95 / base_p95 < constraints.min_latency_reduction - 1e-12:
            reasons.append('latency_reduction')
        result = {'candidate_id': row['candidate_id'], 'p95_ms': p95,
                  'latency_reduction': 1 - p95 / base_p95, 'rejection_reasons': reasons}
        evaluations.append(result)
        if not reasons:
            feasible.append(result)
    winner = min(feasible, key=lambda r: (r['p95_ms'], r['candidate_id'])) if feasible else None
    return {'status': 'candidate_found' if winner else 'no_feasible_candidate',
            'selected': winner, 'candidates': evaluations,
            'next_step': 'freeze candidate and evaluate once on confirm; test is not used here'}
