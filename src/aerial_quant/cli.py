"""Run split auditing or selection from real measurement files."""
import argparse
import json
from pathlib import Path
from .data_audit import audit_splits
from .selection import Constraints, select_measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['audit-splits', 'select'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline', default='fp16')
    parser.add_argument('--constraints')
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding='utf-8'))
    if args.task == 'audit-splits':
        result = audit_splits(data)
    else:
        if not args.constraints:
            parser.error('--constraints is required for select')
        cfg = json.loads(Path(args.constraints).read_text(encoding='utf-8'))
        result = select_measured(data, args.baseline, Constraints(**cfg))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
