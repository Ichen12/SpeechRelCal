"""Original query sampler, uniform-positive loss, and resumable AdamW updates."""
import numpy as np
import torch


def _query_loss(scores, positive, objective='uniform_positive_cross_entropy'):
    return torch.logsumexp(scores, dim=0)-scores[positive].mean()

def train_model(model, queries, probability, config, seed, updates, path):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(seed)
    order, cursor, completed, losses = rng.permutation(len(queries)), 0, 0, []
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['generator_state']
        order, cursor, completed, losses = state['order'], state['cursor'], state['completed_updates'], state['losses']
    def save():
        temporary = path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'generator_state': rng.bit_generator.state, 'order': order, 'cursor': cursor,
                    'completed_updates': completed, 'losses': losses}, temporary)
        temporary.replace(path)
    for update in range(completed, updates):
        terms = []
        for _ in range(config['query_batch_size']):
            if cursor == len(order):
                order, cursor = rng.permutation(len(queries)), 0
            a, r, c, local, valid = queries[int(order[cursor])]; cursor += 1
            p, m = probability('train', a, r, c)
            scores = model(torch.from_numpy(p).to(device), torch.from_numpy(m).to(device))
            scores = scores.clone(); scores[local] = -torch.inf
            terms.append(_query_loss(scores, torch.from_numpy(valid).to(device), 'uniform_positive_cross_entropy'))
        loss = torch.stack(terms).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training objective: {path}, step {update+1}')
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        completed = update+1; losses.append(float(loss.detach()))
        if completed % 100 == 0:
            save()
    save(); model.eval()
    return {'seed': seed, 'training_queries': len(queries), 'updates': completed,
            'initial_loss': losses[0] if losses else None, 'final_loss': losses[-1] if losses else None, 'objective': 'uniform_positive_cross_entropy'}
