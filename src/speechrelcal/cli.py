"""Portable entry points operating on user-prepared scores and queries."""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .calibration import fit_calibration, probability, product
from .metrics import normalized_average_precision
from .models import make_model
from .training import train_model


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_queries(path):
    path = Path(path)
    rows = json.loads(path.read_text())
    for row in rows:
        with np.load(path.parent / row['arrays']) as data:
            row['data'] = {k: data[k] for k in data.files}
        data = row['data']
        # A self-positive or invalid target changes the training objective/nAP.
        positive = data['positive'].astype(bool)
        local = row['self_index']
        if positive[local] or not positive.any() or positive.sum() >= len(positive)-1:
            raise ValueError('Queries need a self-negative, a positive, and a nonself negative.')
        if not np.isfinite(data['inputs']).all():
            raise ValueError('Nonfinite query inputs.')
    return rows


def fit(model, rows, args, updates, checkpoint):
    queries = [(i, {}, np.arange(len(r['data']['positive'])), r['self_index'],
                r['data']['positive'].astype(bool)) for i, r in enumerate(rows)]
    def inputs(role, anchor, relation, candidates):
        d = rows[anchor]['data']
        return d['inputs'].astype(np.float32), d['mask'].astype(np.float32)
    return train_model(model, queries, inputs,
                       dict(learning_rate=args.lr, weight_decay=args.weight_decay, query_batch_size=4),
                       args.seed, updates, checkpoint)


def evaluate(rows, model=None):
    records = []
    for row in rows:
        data = row['data']
        p, m = data['inputs'].astype(np.float32), data['mask'].astype(np.float32)
        if model is None:
            scores = product(p[..., 0] if p.ndim == 3 else p, m)
        else:
            device = next(model.parameters()).device
            with torch.no_grad():
                scores = model(torch.from_numpy(p).to(device), torch.from_numpy(m).to(device)).cpu().numpy()
        keep = np.arange(len(scores)) != row['self_index']
        score = normalized_average_precision(scores[keep], data['positive'][keep])
        records.append(dict(query_id=row['query_id'], anchor_speaker=row['anchor_speaker'],
                            regime=row['regime'], normalized_average_precision=float(score)))
    frame = pd.DataFrame(records)
    aggregate = frame.groupby(['regime', 'anchor_speaker']).normalized_average_precision.mean().groupby('regime').mean()
    return frame, aggregate.to_dict()


def validate_disjoint(fit_rows, val_rows):
    # Prevent validation anchor/candidate leakage into training.
    for key in ('candidate_uids', 'candidate_speakers'):
        left = set().union(*(set(r['data'][key].astype(str)) for r in fit_rows))
        right = set().union(*(set(r['data'][key].astype(str)) for r in val_rows))
        if left & right:
            raise ValueError(f'Training/validation overlap in {key}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('calibrate', help='Fit both interfaces from calibration-only pair CSVs.')
    p.add_argument('--pairs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=2026090581)
    p = commands.add_parser('prepare', help='Map raw query scores with an already fitted calibrator.')
    p.add_argument('--queries', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--interface', choices=['sigmoid','slope_only','intercept_only','full_affine','full_state'], required=True)
    p.add_argument('--relation-aware', action='store_true')
    p.add_argument('--output', required=True)
    for name in ('train', 'select', 'evaluate'):
        p = commands.add_parser(name)
        p.add_argument('--queries', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cpu')
        if name == 'evaluate':
            p.add_argument('--checkpoint', help='Omit for Product.')
        else:
            p.add_argument('--model', choices=['log_linear','deepsets','deepsets_r'], required=True)
            p.add_argument('--seed', type=int, default=2026090411)
            p.add_argument('--weight-decay', type=float, default=0.01)
            if name == 'train':
                p.add_argument('--width', type=int, default=16)
                p.add_argument('--lr', type=float, default=0.001)
                p.add_argument('--epochs', type=int, default=8)
            else:
                p.add_argument('--validation', required=True)
                p.add_argument('--epochs', nargs='+', type=int, default=[8,16,32])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'calibrate':
        frame = pd.read_csv(args.pairs)
        maps = {factor: fit_calibration(group.score.to_numpy(), group.state.to_numpy(),
                                        group.kind.iloc[0], args.seed)
                for factor, group in frame.groupby('factor', sort=False)}
        save_json(args.output, maps)
        return
    if args.command == 'prepare':
        path = Path(args.queries)
        rows = json.loads(path.read_text())
        maps = json.loads(Path(args.calibration).read_text())
        output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            with np.load(path.parent / row['arrays']) as d:
                data = {k:d[k] for k in d.files}
            raw = data.pop('raw'); states = row['states']; factors = row['factors']
            p = np.ones(raw.shape, dtype=np.float32); mask = np.zeros_like(p)
            inputs = np.zeros((*p.shape,3), dtype=np.float32)
            for j, (factor, state) in enumerate(zip(factors, states, strict=True)):
                if not state:
                    continue
                p[:,j] = probability(raw[:,j], maps[factor], state, args.interface)
                mask[:,j] = 1
                order = ('same','different') if factor == 'speaker' else (('slower','faster') if factor == 'rate' else ('lower','higher'))
                inputs[:,j,1+order.index(state)] = 1
            inputs[...,0] = p
            data.update(inputs=inputs if args.relation_aware else p, mask=mask)
            row['arrays'] = f'query_{i}.npz'
            np.savez(output / row['arrays'], **data)
        save_json(output/'queries.json', rows)
        return
    rows = load_queries(args.queries)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    factors = rows[0]['data']['mask'].shape[-1]
    if args.command == 'evaluate':
        model = None
        if args.checkpoint:
            checkpoint = Path(args.checkpoint)
            spec = json.loads(checkpoint.with_suffix('.json').read_text())
            model = make_model(spec['model'], spec['factors'], spec['width']).to(args.device)
            model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=False)['model'])
            model.eval()
        frame, aggregate = evaluate(rows, model)
        frame.to_csv(output/'queries.csv', index=False)
        save_json(output/'metrics.json', aggregate)
    elif args.command == 'train':
        torch.manual_seed(args.seed)
        model = make_model(args.model, factors, args.width).to(args.device)
        audit = fit(model, rows, args, args.epochs*math.ceil(len(rows)/4), output/'model.pt')
        save_json(output/'model.json', dict(model=args.model,factors=factors,width=args.width,
                  lr=args.lr,weight_decay=args.weight_decay,epochs=args.epochs,**audit))
    else:
        validation = load_queries(args.validation)
        validate_disjoint(rows, validation)
        results = []
        widths = [16] if args.model == 'log_linear' else [16,64]
        for width, lr in itertools.product(widths, [0.001,0.003]):
            args.lr = lr
            torch.manual_seed(args.seed)
            model = make_model(args.model, factors, width).to(args.device)
            run = output/f'w{width}_lr{lr}'; run.mkdir(exist_ok=True)
            for epoch in sorted(args.epochs):
                fit(model, rows, args, epoch*math.ceil(len(rows)/4), run/'latest.pt')
                frame, _ = evaluate(validation, model)
                score = float(frame.groupby('anchor_speaker').normalized_average_precision.mean().mean())
                results.append(dict(width=width,lr=lr,epochs=epoch,validation_nap=score))
        save_json(output/'selection.json', dict(model=args.model,seed=args.seed,
                  weight_decay=args.weight_decay,candidates=results,
                  selected=max(results,key=lambda r:r['validation_nap'])))


if __name__ == '__main__':
    main()
