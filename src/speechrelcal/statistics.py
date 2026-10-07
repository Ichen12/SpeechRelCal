"""Paired speaker/source-split/initialization bootstrap used for MSP results.

For a single source split, set source_split_seed to a constant. Published Libri
intervals retain their original seed-specific bootstrap draws in results/source.
"""
import numpy as np
import pandas as pd

def paired(frame, left, right, metric, samples=10000):
    keys=['partition','source_split_seed','seed','query_id','anchor_speaker']
    a=frame[frame.method.eq(left)][keys+[metric]]
    b=frame[frame.method.eq(right)][keys+[metric]]
    pair=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    if len(pair)!=len(a) or len(pair)!=len(b):
        raise ValueError('Methods do not cover identical source splits, model seeds and queries')
    pair['delta']=pair[metric+'_a']-pair[metric+'_b']
    levels=[sorted(pair[k].unique()) for k in ['partition','source_split_seed','seed','anchor_speaker']]
    group=pair.groupby(['partition','source_split_seed','seed','anchor_speaker']).delta.agg(['sum','count'])
    group=group.reindex(pd.MultiIndex.from_product(levels,names=group.index.names),fill_value=0)
    shape=tuple(map(len,levels));sums=group['sum'].to_numpy().reshape(shape);counts=group['count'].to_numpy().reshape(shape)
    point=sums.sum((0,1,2))/counts.sum((0,1,2));rng=np.random.default_rng(2026090904)
    speaker_draw=rng.integers(shape[3],size=(samples,shape[3]))
    result=dict(estimate=float(point.mean()),speaker_count=shape[3],partition_count=shape[0],
        source_split_count=shape[1],initialization_seed_count=shape[2],
        speaker_only_ci=np.quantile(point[speaker_draw].mean(1),[.025,.975]).tolist())
    for mode in ['shared_seed_across_fixed_partitions','partition_and_seed_sensitivity']:
        draws=[]
        for start in range(0,samples,100):
            n=min(100,samples-start)
            p=(np.broadcast_to(np.arange(shape[0]),(n,shape[0])) if mode.startswith('shared')
               else rng.integers(shape[0],size=(n,shape[0])))
            s=rng.integers(shape[1],size=(n,shape[1]));i=rng.integers(shape[2],size=(n,shape[2]))
            index=(p[:,:,None,None],s[:,None,:,None],i[:,None,None,:])
            values=sums[index].sum((1,2,3))/counts[index].sum((1,2,3))
            draws.extend(np.take_along_axis(values,speaker_draw[start:start+n],axis=1).mean(1))
        result[mode]=np.quantile(draws,[.025,.975]).tolist()
    return result
