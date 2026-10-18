"""Causal image-space pair features and a grouped Random Forest training pipeline.
No risk labels are generated from geometric thresholds. No metres/km/h/TTC claims.
"""
from pathlib import Path
import json
import pickle
import hashlib
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             balanced_accuracy_score, cohen_kappa_score, accuracy_score)

STATES = ['Normal', 'Increasing Risk', 'Developing Conflict', 'Critical Conflict / Collision']
FEATURES = ['separation_min','separation_mean','separation_last','closing_mean','closing_max',
            'relative_speed_mean','speed_a_mean','speed_b_mean','speed_change_a',
            'speed_change_b','direction_alignment','bbox_iou_max','approach_fraction',
            'separation_change','coverage']
SCHEMA = 'image_pair_v1'

def pair_features(tracks, video_id, window=5., stride=1., sample_hz=5., min_coverage=.8):
    """Windows end at t: all features use observations <= t. Units: frame-normalized/s.
    A pair must be observed together for >=80% of the five-second window; missing
    observations are not interpolated. Canonical ID order makes training reproducible.
    """
    from itertools import combinations
    output = []
    if tracks.empty:
        return pd.DataFrame()
    tracks = tracks.sort_values(['time','track_id']).drop_duplicates(['time','track_id'])
    end = float(tracks.time.max())
    for finish in np.arange(window, end + 1e-6, stride):
        start = finish - window
        clip = tracks[(tracks.time > start) & (tracks.time <= finish)]
        for a,b in combinations(sorted(clip.track_id.unique()), 2):
            x = clip[clip.track_id == a].merge(clip[clip.track_id == b], on='time', suffixes=('_a','_b')).sort_values('time')
            coverage = min(len(x)/(window*sample_hz), 1.)
            if len(x)<8 or coverage<min_coverage or x.time.max()-x.time.min()<window*.7:
                continue
            t=x.time.to_numpy()
            # Bottom centre is a rough road contact point, normalized by frame size.
            pa=x[['x_a','y_a']].to_numpy(); pb=x[['x_b','y_b']].to_numpy()
            # Trailing smoothing: no future samples are used.
            pa=pd.DataFrame(pa).rolling(3,min_periods=1).mean().to_numpy()
            pb=pd.DataFrame(pb).rolling(3,min_periods=1).mean().to_numpy()
            dt=np.diff(t)
            # Reject intervals with a substantial tracking gap.
            valid=(dt>0)&(dt<=2.1/sample_hz)
            if valid.sum()<5:
                continue
            va=np.diff(pa,axis=0)[valid]/dt[valid,None]
            vb=np.diff(pb,axis=0)[valid]/dt[valid,None]
            sa=np.linalg.norm(va,axis=1); sb=np.linalg.norm(vb,axis=1)
            sep=np.linalg.norm(pa-pb,axis=1)
            closing=-np.diff(sep)[valid]/dt[valid]
            alignment=np.sum(va*vb,axis=1)/(sa*sb+1e-8)
            ix=np.maximum(0,np.minimum(x.x2_a,x.x2_b)-np.maximum(x.x1_a,x.x1_b))
            iy=np.maximum(0,np.minimum(x.y2_a,x.y2_b)-np.maximum(x.y1_a,x.y1_b))
            inter=ix*iy
            area_a=(x.x2_a-x.x1_a)*(x.y2_a-x.y1_a)
            area_b=(x.x2_b-x.x1_b)*(x.y2_b-x.y1_b)
            iou=inter/(area_a+area_b-inter+1e-8)
            values=[sep.min(),sep.mean(),sep[-1],closing.mean(),closing.max(),
                    np.linalg.norm(va-vb,axis=1).mean(),sa.mean(),sb.mean(),
                    sa[-1]-sa[0],sb[-1]-sb[0],alignment.mean(),iou.max(),
                    (closing>0).mean(),sep[-1]-sep[0],coverage]
            row=dict(zip(FEATURES,map(float,values)))
            row.update(window_id=f'{video_id}__{int(a)}_{int(b)}__{finish:.3f}',
                       video_id=video_id,pair_a=int(a),pair_b=int(b),start_s=float(start),end_s=float(finish))
            output.append(row)
    return pd.DataFrame(output)

def train_model(dataset, output_dir, test_groups):
    """Explicit held-out source recordings; no test-driven parameter selection."""
    df=dataset.copy()
    required=FEATURES+['label','source_group','video_id','window_id','evidence']
    absent=set(required)-set(df.columns)
    if absent:
        raise ValueError(f'Missing columns: {sorted(absent)}')
    if df.window_id.duplicated().any():
        raise ValueError('Duplicate window IDs. Keep one reviewed annotation per window.')
    if df.source_group.isna().any() or df.source_group.astype(str).str.strip().eq('').any():
        raise ValueError('Every row needs a source_group for the original recording.')
    if (df.groupby('video_id').source_group.nunique()>1).any():
        raise ValueError('A video cannot belong to multiple source groups.')
    if not df.label.isin(STATES).all():
        raise ValueError('Unexpected risk label.')
    if df.evidence.fillna('').astype(str).str.strip().str.len().lt(10).any():
        raise ValueError('Add an evidence note (at least 10 characters) for every annotation.')
    df.source_group=df.source_group.astype(str)
    test_groups=set(map(str,test_groups))
    if not test_groups or not test_groups.issubset(set(df.source_group)):
        raise ValueError('Select valid held-out source groups.')
    te=df.source_group.isin(test_groups); train=df[~te]; test=df[te]
    if train.source_group.nunique()<2 or test.source_group.nunique()<1:
        raise ValueError('Need >=2 training recordings and >=1 held-out recording.')
    for name,part in [('training',train),('test',test)]:
        missing=set(STATES)-set(part.label)
        if missing:
            raise ValueError(f'{name} is missing states: {sorted(missing)}. Annotate more independent recordings or change the predefined split; do not claim four-state performance yet.')
    X=df[FEATURES].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Features contain non-finite values; re-extract the affected videos.')
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),
                    ('rf',RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=2,
                                                  class_weight='balanced',random_state=42,n_jobs=-1))])
    model.fit(X[~te],train.label)
    pred=model.predict(X[te]); probs=model.predict_proba(X[te])
    majority=train.label.value_counts().idxmax()
    report={'schema':SCHEMA,'random_state':42,'train_groups':sorted(set(train.source_group)),
            'test_groups':sorted(test_groups),'train_windows':len(train),'test_windows':len(test),
            'train_class_counts':train.label.value_counts().to_dict(),'test_class_counts':test.label.value_counts().to_dict(),
            'accuracy':accuracy_score(test.label,pred),'macro_f1':f1_score(test.label,pred,labels=STATES,average='macro',zero_division=0),
            'balanced_accuracy':balanced_accuracy_score(test.label,pred),
            'quadratic_weighted_kappa':cohen_kappa_score(test.label,pred,labels=STATES,weights='quadratic'),
            'majority_baseline_macro_f1':f1_score(test.label,[majority]*len(test),labels=STATES,average='macro',zero_division=0),
            'classification_report':classification_report(test.label,pred,labels=STATES,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(test.label,pred,labels=STATES).tolist(),
            'states':STATES,'probabilities':'Uncalibrated Random Forest vote proportions; not collision probabilities.',
            'limitations':['Overlapping windows are correlated; counts are not independent event counts.',
                           'Pilot image-space model; perspective and tracker errors affect predictions.',
                           'No validated early-warning lead time, physical TTC or geographic generalization claim.'],
            'python':platform.python_version(),'sklearn':sklearn.__version__,
            'dataset_sha256':hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    bundle={'schema':SCHEMA,'model':model,'features':FEATURES,'states':STATES,'report':report}
    with (out/'risk_model.pkl').open('wb') as f: pickle.dump(bundle,f)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    df[['window_id','video_id','source_group','label']].assign(split=np.where(te,'test','train')).to_csv(out/'split_manifest.csv',index=False)
    results=test[['window_id','video_id','pair_a','pair_b','end_s','label']].copy()
    results['prediction']=pred; results['vote_share']=probs.max(axis=1)
    results.to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame({'feature':FEATURES,'importance':model.named_steps['rf'].feature_importances_}).sort_values('importance',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    return report

def load_model(path):
    # Pickle can execute code. Caller accepts only a locally trained/trusted artifact.
    with open(path,'rb') as f: bundle=pickle.load(f)
    if bundle.get('schema')!=SCHEMA or bundle.get('features')!=FEATURES:
        raise ValueError('Incompatible feature schema; retrain with this package.')
    return bundle

def predict_windows(bundle, windows):
    out=windows.copy()
    if out.empty: return out
    probabilities=bundle['model'].predict_proba(out[FEATURES])
    out['prediction']=bundle['model'].classes_[probabilities.argmax(axis=1)]
    out['vote_share']=probabilities.max(axis=1)
    return out
