"""Check released aggregates against the user-supplied final paper (no data needed)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = 'normalized_average_precision'
REGIMES = ['heldout_pair_tuple','zero_shot_triple','zero_shot_quadruple']


def check():
    ref = json.loads((ROOT/'results/paper_reference.json').read_text())
    table = pd.read_csv(ROOT/'results/table1.csv')
    figures = {k:pd.read_csv(ROOT/f'results/{k}.csv') for k in ['fig2','fig3a','fig3b']}
    discrepancies = []
    counts = dict(table1_values=0,fig3a_values=0,fig3a_significance_marks=0,source_estimates=0,source_intervals=0)
    def equal(a,b,tolerance=1e-10):
        if not np.isclose(a,b,atol=tolerance,rtol=0):
            raise AssertionError(f'{a} != {b}')
    for method, expected in ref['table1'].items():
        interface, composer = method.split('_',1)
        for column, value in zip(ref['table1_columns'],expected,strict=True):
            dataset,arity = column.split('-');arity=int(arity)
            row=table[table.dataset.eq(dataset)&table.attributes.eq(arity)&table.interface.eq(interface)&table.composer.eq(composer)].iloc[0]
            equal(round(row.nap_percent,2),value); counts['table1_values']+=1
            original=pd.read_csv(ROOT/f'results/source/table1_{dataset}_speaker_equal_aggregate.csv')
            source=original[original.method.eq(method)&original.regime.eq(REGIMES[arity-2])].iloc[0]
            equal(row.nap_percent,source[M]*100);counts['source_estimates']+=1
    for method,matrix in ref['fig3a'].items():
        for i,column in enumerate(ref['table1_columns']):
            dataset,arity=column.split('-');arity=int(arity)
            for j,budget in enumerate(ref['fig3a_budgets']):
                f=figures['fig3a'];r=f[f.dataset.eq(dataset)&f.attributes.eq(arity)&f.method.eq(method)&f.budget.eq(budget)].iloc[0]
                equal(round(r.delta_nap_pp,2),matrix[i][j]);counts['fig3a_values']+=1
                significant=bool(r.ci_low_pp>0 or r.ci_high_pp<0)
                if significant != ref['fig3a_stars'][method][i][j]:
                    discrepancies.append(dict(figure='3(a)',model=method,setting=column,budget=budget,paper_star=ref['fig3a_stars'][method][i][j],source_significant=significant,ci_pp=[r.ci_low_pp,r.ci_high_pp]))
                counts['fig3a_significance_marks']+=1
    for figure,frame in figures.items():
        for row in frame.itertuples():
            source=json.loads((ROOT/f'results/source/{figure}_{row.dataset}_paired_uncertainty.json').read_text())
            if figure=='fig2':
                prefix='heldout_' if row.dataset=='msp' else ''
                left,right=prefix+row.method,prefix+'sigmoid'
            elif figure=='fig3a':left,right=f'{row.budget:.2f}_{row.method}','product'
            else:left,right='matched_relation_deepsets','product'
            r=next(r for r in source if r['left_method']==left and r['right_method']==right and r['regime']==REGIMES[row.attributes-2] and r['metric']==M)
            equal(row.delta_nap_pp,r['estimate']*100);counts['source_estimates']+=1
            for val,expected in zip([row.ci_low_pp,row.ci_high_pp,row.sensitivity_low_pp,row.sensitivity_high_pp],r['shared_seed_across_fixed_partitions']+r['partition_and_seed_sensitivity'],strict=True):
                equal(val,expected*100);counts['source_intervals']+=1
            # Confirm CI point estimates use the same speaker-equal aggregate as tables.
            agg=pd.read_csv(ROOT/f'results/source/{figure}_{row.dataset}_speaker_equal_aggregate.csv')
            sub=agg[agg.regime.eq(REGIMES[row.attributes-2])].set_index('method')
            equal(row.delta_nap_pp,(sub.loc[left,M]-sub.loc[right,M])*100,1e-8)
    fig2=figures['fig2']; full=fig2[fig2.method.eq('full_affine')]
    equal(round(full.delta_nap_pp.min(),2),5.03);equal(round(full.delta_nap_pp.max(),2),15.29)
    pivot=fig2.pivot(index=['dataset','attributes'],columns='method',values='delta_nap_pp')
    recovery=100*pivot.intercept_only/pivot.full_affine
    equal(round(recovery.min(),1),84.6);equal(round(recovery.max(),1),98.5)
    assert (full.ci_low_pp>0).all() and (full.sensitivity_low_pp>0).all()
    atomic={}
    for dataset in ['msp','libri']:
        f=pd.read_csv(ROOT/f'results/source/fig2_{dataset}_atomic.csv')
        if dataset=='msp':f=f[f.arm.eq('heldout')]
        stats=f.groupby('method')[['ece','nll','brier']].mean()
        target=(.254,.020) if dataset=='msp' else (.278,.006)
        equal(round(stats.loc['sigmoid','ece'],3),target[0]);equal(round(stats.loc['full_affine','ece'],3),target[1])
        assert (stats.loc['full_affine',['nll','brier']]<stats.loc['sigmoid',['nll','brier']]).all()
        atomic[dataset]=stats.loc[['sigmoid','full_affine']].to_dict('index')
    auroc=pd.read_csv(ROOT/'results/source/full_state_auroc.csv')
    assert len(auroc)==30
    maximum=float(auroc.delta_fp32.abs().max())
    equal(float(f'{maximum:.1e}'),3.6e-8,1e-20)
    prior=json.loads((ROOT/'results/source/prior_findings.json').read_text())['intercept_recovery_range']
    equal(round(prior['min']*100,1),82.1);equal(round(prior['max']*100,1),98.5)
    matched=figures['fig3b'];libri=matched[matched.dataset.eq('libri')]
    assert (libri.delta_nap_pp.round(2)==.55).all() and (libri.sensitivity_low_pp>0).all()
    assert (matched[matched.dataset.eq('msp')].delta_nap_pp<0).all()
    # Additional numerical statements in Sections 4.3–4.4.
    t=table.pivot(index=['dataset','attributes'],columns=['interface','composer'],values='nap_percent')
    gap=(t['calibrated','deepsets']-t['calibrated','product']).abs().max()
    equal(round(gap,2),.21)
    difference=t['calibrated','product']-t['standardized','deepsets']
    equal(round(difference.min(),2),1.47);equal(round(difference.max(),2),4.75)
    return dict(status='numerical_values_match_with_annotation_difference' if discrepancies else 'passed',discrepancies=discrepancies,paper_sha256=ref['pdf_sha256'],checks=counts,
                full_affine_gain_range_pp=[float(full.delta_nap_pp.min()),float(full.delta_nap_pp.max())],
                intercept_recovery_percent=[float(recovery.min()),float(recovery.max())],
                atomic_metrics=atomic,max_full_state_auroc_change=maximum,
                scope='Final-paper rounded numerical values and aggregate-source consistency; no audio experiments rerun.',
                implementation_notes=['Ordinal historical code uses inclusive >=/<= thresholds; paper notation uses strict >/<. See docs/reproduction.md.',
                                      'Full-state ordinal calibration retains sklearn default L2 regularization; event affine BCE is unregularized.'])


if __name__=='__main__':
    report=check()
    text=json.dumps(report,indent=2)+'\n'
    (ROOT/'results/paper_check.json').write_text(text)
    print(text)
