import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ARMS = ('F11', 'F01', 'F10', 'F00')
CONTRASTS = {'H1': ('F11', 'F01'), 'H2': ('F11', 'F10'), 'H3': ('F11', 'F00')}
PROB = ['p_akiec','p_bcc','p_bkl','p_df','p_mel','p_nv','p_vasc']
K = 7

def load_inputs(root):
    data = {}
    base = None
    hashes = {}
    for arm in ARMS:
        data[arm] = []
        for seed in range(3):
            p = root / f'{arm}_s{seed}' / 'e1.csv'
            if not p.is_file():
                raise FileNotFoundError(p)
            hashes[str(p)] = hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
            d = pd.read_csv(p, dtype={'cluster_id':str})
            for col in ['image_id','cluster_id','label_idx',*PROB]:
                if col not in d.columns:
                    raise ValueError(f'{p}: missing {col}')
            if base is None:
                base = d[['image_id','cluster_id','label_idx']].copy()
                if base.image_id.duplicated().any():
                    raise ValueError('Duplicate image IDs in reference')
            elif not d[['image_id','cluster_id','label_idx']].equals(base):
                raise ValueError(f'{p}: image order, cluster, or label mismatch')
            data[arm].append(d[PROB].to_numpy(dtype=np.float64).argmax(axis=1).astype(np.int8))
    return base, data, hashes

def cluster_layout(base):

    codes, uniques = pd.factorize(base['cluster_id'], sort=False)
    y = base.label_idx.to_numpy(dtype=np.int64)

    clabel = base.groupby(codes)['label_idx'].agg(lambda s: s.value_counts().index[0]).to_numpy(dtype=int)
    return y, codes, len(uniques), clabel

def bma_from_confusions(c):

    support = c.sum(axis=2)
    diagonal = np.diagonal(c, axis1=1, axis2=2)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support>0)
    return np.divide((recall*(support>0)).sum(axis=1), (support>0).sum(axis=1),
                     out=np.zeros(len(c), dtype=np.float64), where=(support>0).sum(axis=1)>0)

def compute(root, outdir, B, seed, chunk):
    base, predictions, hashes = load_inputs(root)
    y, codes, nc, clabel = cluster_layout(base)
    n = len(y)
    if n != 14268:
        raise ValueError(f'E1 size unexpected: {n}')
    rng = np.random.default_rng(seed)


    mult = np.zeros((B, nc), dtype=np.float32)
    for c in np.unique(clabel):
        members = np.flatnonzero(clabel == c)
        draws = rng.integers(0, len(members), size=(B, len(members)))
        np.add.at(mult, (np.repeat(np.arange(B), len(members)), members[draws].ravel()), 1.0)

    cluster_matrices = {}
    point = {}
    for arm, ps in predictions.items():
        mat = []
        vals = []
        for pred in ps:
            index = y*K + pred
            cm = np.zeros((nc, K*K), dtype=np.float32)
            np.add.at(cm, (codes, index), 1.0)
            mat.append(cm)
            raw = np.bincount(index, minlength=K*K).reshape(1,K,K).astype(np.float64)
            vals.append(float(bma_from_confusions(raw)[0]))
        cluster_matrices[arm] = mat
        point[arm] = {'per_seed': vals, 'mean': float(np.mean(vals))}
    bootstrap = {arm: np.empty(B, dtype=np.float64) for arm in ARMS}
    for lo in range(0, B, chunk):
        hi = min(B, lo+chunk)
        m = mult[lo:hi].astype(np.float64)
        for arm in ARMS:
            per_seed = []
            for cm in cluster_matrices[arm]:
                c = (m @ cm.astype(np.float64)).reshape(-1,K,K)
                per_seed.append(bma_from_confusions(c))
            bootstrap[arm][lo:hi] = np.mean(per_seed, axis=0)
    results = {}
    for name,(a,b) in CONTRASTS.items():
        d = bootstrap[a]-bootstrap[b]
        p = min(1., 2*min(np.mean(d<=0),np.mean(d>=0)))
        results[name] = {'arms':[a,b], 'delta':point[a]['mean']-point[b]['mean'],
                         'ci95':[float(x) for x in np.percentile(d,[2.5,97.5])],
                         'p_boot':float(p), 'negative_or_zero_replicates':int((d<=0).sum()),
                         'positive_or_zero_replicates':int((d>=0).sum())}
    running = 0.
    for i,(name,r) in enumerate(sorted(results.items(), key=lambda kv:kv[1]['p_boot'])):
        running = max(running, min(1.,(len(results)-i)*r['p_boot']))
        r['p_holm'] = running
    report = {'method':'Independent cluster-level confusion bootstrap; no skinclip imports',
              'seed':seed,'n_boot':B,'n_images':n,'n_clusters':nc,
              'n_seed_runs':12,'point_estimates':point,'confirmatory':results,
              'input_sha256':hashes,
              'note':'Resampling matches the documented majority-class stratification and NumPy draw ordering. Results independently computed from original per-run CSVs.'}
    outdir.mkdir(parents=True,exist_ok=True)
    path = outdir/'independent_bootstrap.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))
    print('Report:',path)

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--eval-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--n-boot',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=12345)
    ap.add_argument('--chunk',type=int,default=100)
    a=ap.parse_args()
    compute(a.eval_root,a.output_dir,a.n_boot,a.seed,a.chunk)
