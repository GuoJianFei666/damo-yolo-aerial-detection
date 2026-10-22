"""Use stable module names instead of hard-coded enumeration indices."""
import hashlib
import json
from typing import Iterable


def module_names_hash(available_names: Iterable[str]) -> str:
    names = list(available_names)
    if not names or len(names) != len(set(names)) or any(not n for n in names):
        raise ValueError('available module names must be nonempty and unique')
    return hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest()


def resolve_policy(policy: dict, available_names: Iterable[str]) -> list[str]:
    names = list(available_names)
    fingerprint = module_names_hash(names)
    if policy.get('schema_version') != 1:
        raise ValueError('unsupported policy schema')
    if policy.get('module_names_sha256') != fingerprint:
        raise ValueError('policy does not match the quantizable module names')
    requested = policy.get('quantize')
    if not isinstance(requested, list) or any(not isinstance(n, str) for n in requested):
        raise ValueError('quantize must be an explicit list of names')
    if len(requested) != len(set(requested)):
        raise ValueError('duplicate names in policy')
    unknown = set(requested) - set(names)
    if unknown:
        raise ValueError(f'unknown module names: {sorted(unknown)}')
    return sorted(requested)


def load_policy(path: str, available_names: Iterable[str]) -> list[str]:
    with open(path, encoding='utf-8') as stream:
        return resolve_policy(json.load(stream), available_names)


def conv2d_constructor_kwargs(conv) -> dict:
    """Preserve Conv2d geometry for a compatible quantized-convolution factory.

    Tested with torch.nn.Conv2d only. NVIDIA QuantConv2d integration still needs
    verification in the pinned upstream environment. Weights must be copied by
    the caller and the resulting model must pass numerical equivalence checks.
    """
    return {name: getattr(conv, name) for name in
            ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
             'dilation', 'groups', 'padding_mode')} | {'bias': conv.bias is not None}
