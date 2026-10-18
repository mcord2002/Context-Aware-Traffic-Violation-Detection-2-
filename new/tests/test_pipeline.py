"""Artificial fixtures test software only; no research results are produced."""
import unittest
from pathlib import Path
import tempfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from risk_core import pair_features, FEATURES, STATES, train_model, load_model, predict_windows

def tracks():
    rows=[]
    for t in np.arange(0,8.01,.2):
        for tid in [1,2]:
            x=.1+.03*t if tid==1 else .8-.02*t
            rows.append([round(t,3),tid,2,x,.6,x-.04,.5,x+.04,.6])
    return pd.DataFrame(rows,columns=['time','track_id','class_id','x','y','x1','y1','x2','y2'])

def dataset():
    rng=np.random.default_rng(3); rows=[]
    for group in range(3):
        for state in range(4):
            for i in range(8):
                r=dict(zip(FEATURES,rng.normal(state, .4,len(FEATURES))))
                r.update(window_id=f'g{group}_s{state}_{i}',video_id=f'video{group}',source_group=f'group{group}',
                         pair_a=1,pair_b=2,end_s=5+i,label=STATES[state],evidence='Artificial software fixture only')
                rows.append(r)
    return pd.DataFrame(rows)

class PipelineTests(unittest.TestCase):
    def test_features_and_causality(self):
        t=tracks(); f=pair_features(t,'fixture')
        self.assertEqual(len(f),4)
        self.assertTrue(np.isfinite(f[FEATURES].to_numpy()).all())
        self.assertGreater(f.iloc[0].closing_mean,0)
        # Replacing future observations must not alter the window ending at 5s.
        altered=t.copy(); altered.loc[altered.time>5,'x']=99
        g=pair_features(altered,'fixture')
        np.testing.assert_allclose(f.iloc[0][FEATURES].astype(float),g.iloc[0][FEATURES].astype(float))
    def test_sparse_pair_excluded(self):
        t=tracks(); t=t[~((t.track_id==2)&(t.time>1))]
        self.assertTrue(pair_features(t,'fixture').empty)
    def test_group_split_and_reload(self):
        d=dataset()
        with tempfile.TemporaryDirectory() as folder:
            r=train_model(d,folder,['group2'])
            self.assertFalse(set(r['train_groups'])&set(r['test_groups']))
            bundle=load_model(Path(folder)/'risk_model.pkl')
            predicted=predict_windows(bundle,d)
            self.assertEqual(len(predicted),len(d))
            self.assertTrue(predicted.prediction.isin(STATES).all())
            self.assertTrue(predicted.vote_share.between(0,1).all())
            manifest=pd.read_csv(Path(folder)/'split_manifest.csv')
            self.assertTrue((manifest.groupby('source_group').split.nunique()==1).all())
    def test_missing_state_rejected(self):
        d=dataset(); d=d[~((d.source_group=='group2')&(d.label==STATES[-1]))]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'test is missing'): train_model(d,folder,['group2'])
    def test_duplicate_rejected(self):
        d=dataset(); d=pd.concat([d,d.iloc[:1]],ignore_index=True)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'Duplicate'): train_model(d,folder,['group2'])
    def test_extraction_contract_with_mock_detector(self):
        # Fake detector validates integration/CSV contracts, not detection accuracy.
        import cv2
        from unittest.mock import patch
        from types import SimpleNamespace
        from video_pipeline import extract_video
        class Tensor:
            def __init__(self, values): self.values=np.array(values)
            def cpu(self): return self
            def numpy(self): return self.values
        class Detector:
            def __init__(self, weights): self.count=0
            def track(self, frame, **kwargs):
                self.count+=1
                boxes=SimpleNamespace(xyxy=Tensor([[30+self.count,80,60+self.count,120],[240-self.count,80,270-self.count,120]]),
                                      id=Tensor([1,2]),cls=Tensor([2,2]))
                return [SimpleNamespace(boxes=boxes)]
        with tempfile.TemporaryDirectory() as folder:
            video=Path(folder)/'fixture.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,180))
            self.assertTrue(writer.isOpened())
            for _ in range(81): writer.write(np.zeros((180,320,3),dtype=np.uint8))
            writer.release()
            with patch.dict(sys.modules, {'ultralytics':SimpleNamespace(YOLO=Detector)}):
                track,features,meta,out=extract_video(video,Path(folder)/'runs',max_seconds=8)
            self.assertGreater(len(features),0)
            self.assertTrue((out/'tracks.csv').exists())
            self.assertTrue((out/'features.csv').exists())
            self.assertEqual(meta['sample_hz'],5)
            self.assertTrue(track.x.between(0,1).all())
    def test_batch_deduplication_cache_and_failure(self):
        from batch import save_uploads,extract_batch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            uploads=[SimpleNamespace(name='a.mp4',getvalue=lambda:b'a'),
                     SimpleNamespace(name='copy.mp4',getvalue=lambda:b'a'),
                     SimpleNamespace(name='b.mp4',getvalue=lambda:b'b'),
                     SimpleNamespace(name='c.mp4',getvalue=lambda:b'c')]
            videos=save_uploads(uploads,Path(folder)/'videos')
            self.assertEqual(len(videos),3)
            runs=Path(folder)/'runs'; cached=runs/videos[0]['video_id']; cached.mkdir(parents=True)
            (cached/'features.csv').touch(); (cached/'metadata.json').touch()
            called=[]; progress=[]
            def extractor(path,*args,**kwargs):
                called.append(path)
                if path==videos[1]['path']: raise ValueError('fixture failure')
            result=extract_batch(videos,runs,extractor,'fixture.pt',30,progress.append)
            self.assertEqual(len(called),2)
            self.assertTrue(result[0]['status'].startswith('Cached'))
            self.assertEqual(result[1]['status'],'Failed')
            self.assertEqual(result[2]['status'],'Extracted')
            self.assertEqual(progress[-1],1.)
    def test_video_helpers(self):
        import cv2
        from video_pipeline import preview_pair,render_predictions
        with tempfile.TemporaryDirectory() as folder:
            video=Path(folder)/'fixture.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,180))
            self.assertTrue(writer.isOpened())
            for _ in range(81): writer.write(np.zeros((180,320,3),dtype=np.uint8))
            writer.release()
            t=tracks(); f=pair_features(t,'fixture')
            sheet=preview_pair(video,t,f.iloc[0]); self.assertEqual(sheet.shape,(540,1440,3))
            f['prediction']=STATES[0]; f['vote_share']=.7
            rendered=render_predictions(video,t,f,Path(folder)/'annotated.mp4')
            cap=cv2.VideoCapture(str(rendered)); ok,_=cap.read(); cap.release()
            self.assertTrue(ok)

if __name__=='__main__': unittest.main()
