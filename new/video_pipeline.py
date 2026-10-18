"""YOLO + ByteTrack extraction. Detector/tracker is support, not the risk model."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from risk_core import pair_features

TRACK_COLUMNS=['time','track_id','class_id','x','y','x1','y1','x2','y2']

def extract_video(video_path, out_dir, weights='yolov8n.pt', sample_hz=5., max_seconds=120., callback=None):
    import cv2
    from ultralytics import YOLO
    source=Path(video_path)
    digest=hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    video_id=digest.hexdigest()[:16]
    out=Path(out_dir)/video_id; out.mkdir(parents=True,exist_ok=True)
    # A fresh tracker for every video prevents IDs leaking between sources.
    detector=YOLO(weights)
    cap=cv2.VideoCapture(str(source))
    fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or not np.isfinite(fps) or fps<=0:
        cap.release(); raise ValueError('Cannot read video/FPS. Convert to MP4 H.264 and retry.')
    step=max(1,int(round(fps/sample_hz))); actual_hz=fps/step
    limit=min(n,int(max_seconds*fps)) if n>0 else int(max_seconds*fps)
    rows=[]; index=0; began=time.perf_counter()
    try:
        while index<limit:
            ok,frame=cap.read()
            if not ok: break
            if index%step==0:
                h,w=frame.shape[:2]
                result=detector.track(frame,persist=True,tracker='bytetrack.yaml',classes=[2,3,5,7],conf=.25,verbose=False)[0]
                boxes=result.boxes
                if boxes is not None and boxes.id is not None:
                    for xyxy,tid,cls in zip(boxes.xyxy.cpu().numpy(),boxes.id.cpu().numpy(),boxes.cls.cpu().numpy()):
                        x1,y1,x2,y2=xyxy/[w,h,w,h]
                        rows.append([index/fps,int(tid),int(cls),(x1+x2)/2,y2,x1,y1,x2,y2])
                if callback: callback(min((index+1)/max(limit,1),1.))
            index+=1
    finally: cap.release()
    tracks=pd.DataFrame(rows,columns=TRACK_COLUMNS)
    features=pair_features(tracks,video_id,sample_hz=actual_hz)
    # Keep a schema even when no qualifying pair exists.
    if features.empty:
        from risk_core import FEATURES
        features=pd.DataFrame(columns=['window_id','video_id','pair_a','pair_b','start_s','end_s']+FEATURES)
    tracks.to_csv(out/'tracks.csv',index=False); features.to_csv(out/'features.csv',index=False)
    metadata={'video_id':video_id,'source_sha256':digest.hexdigest(),'source_name':source.name,
              'fps':fps,'sample_hz':actual_hz,'processed_seconds':index/fps,'elapsed_seconds':time.perf_counter()-began,
              'weights':weights,'units':'normalized image coordinates; no physical calibration',
              'classes':'COCO car/motorcycle/bus/truck; three-wheelers may be missed/misclassified'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return tracks,features,metadata,out

def preview_pair(video_path, tracks, row):
    """Contact sheet with the exact target IDs, for evidence-based annotation."""
    import cv2
    cap=cv2.VideoCapture(str(video_path)); pictures=[]
    try:
        for timestamp in np.linspace(float(row.start_s)+.2,float(row.end_s),6):
            cap.set(cv2.CAP_PROP_POS_MSEC,timestamp*1000); ok,frame=cap.read()
            if not ok: continue
            h,w=frame.shape[:2]
            nearby=tracks.iloc[(tracks.time-timestamp).abs().argsort()[:1]]
            if not nearby.empty and abs(float(nearby.time.iloc[0])-timestamp)<.5:
                observed=tracks[tracks.time==nearby.time.iloc[0]]
                for r in observed.itertuples():
                    if r.track_id not in [row.pair_a,row.pair_b]: continue
                    p1=(int(r.x1*w),int(r.y1*h)); p2=(int(r.x2*w),int(r.y2*h))
                    cv2.rectangle(frame,p1,p2,(0,220,255),3)
                    cv2.putText(frame,f'ID {r.track_id}',p1,cv2.FONT_HERSHEY_SIMPLEX,.8,(0,220,255),2)
            cv2.putText(frame,f'{timestamp:.1f}s',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            pictures.append(cv2.resize(frame,(480,270)))
    finally: cap.release()
    if len(pictures)!=6: return None
    return cv2.cvtColor(np.vstack([np.hstack(pictures[:3]),np.hstack(pictures[3:])]),cv2.COLOR_BGR2RGB)

def render_predictions(video_path, tracks, predictions, destination):
    """Offline annotated MP4; overlays use the latest completed window only."""
    import cv2
    import shutil
    import subprocess
    cap=cv2.VideoCapture(str(video_path)); fps=cap.get(cv2.CAP_PROP_FPS)
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw=Path(destination).with_suffix('.raw.mp4')
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened(): cap.release(); raise ValueError('Video writer unavailable.')
    last_time=float(tracks.time.max()) if not tracks.empty else 0
    # Index by sample and prediction time: avoid re-scanning the entire table per frame.
    samples={float(t):v for t,v in tracks.groupby('time')}; times=np.array(sorted(samples))
    ptimes=np.array(sorted(predictions.end_s.unique())) if not predictions.empty else np.array([])
    pmap={float(t):v for t,v in predictions.groupby('end_s')} if not predictions.empty else {}
    frame_id=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok or frame_id/fps>last_time: break
            t=frame_id/fps; visible=set()
            k=np.searchsorted(times,t,side='right')-1
            if k>=0 and t-times[k]<.35:
                for r in samples[times[k]].itertuples():
                    visible.add(r.track_id)
                    cv2.rectangle(frame,(int(r.x1*w),int(r.y1*h)),(int(r.x2*w),int(r.y2*h)),(255,170,0),2)
                    cv2.putText(frame,f'ID {r.track_id}',(int(r.x1*w),max(15,int(r.y1*h)-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,170,0),2)
            j=np.searchsorted(ptimes,t,side='right')-1
            text='Collecting 5-second history / no eligible pair'
            if j>=0 and t-ptimes[j]<1.1:
                candidates=pmap[ptimes[j]]
                candidates=candidates[candidates.pair_a.isin(visible)&candidates.pair_b.isin(visible)]
                if not candidates.empty:
                    from risk_core import STATES
                    best=candidates.assign(rank=candidates.prediction.map({s:i for i,s in enumerate(STATES)})).sort_values(['rank','vote_share'],ascending=False).iloc[0]
                    text=f'Pair {best.pair_a}-{best.pair_b}: {best.prediction} ({best.vote_share:.0%} votes)'
            cv2.rectangle(frame,(0,0),(w,45),(0,0,0),-1)
            cv2.putText(frame,text,(10,29),cv2.FONT_HERSHEY_SIMPLEX,max(.35,min(.65,w/1300)),(255,255,255),1)
            writer.write(frame); frame_id+=1
    finally: writer.release(); cap.release()
    destination=Path(destination)
    if shutil.which('ffmpeg'):
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)],check=True)
        raw.unlink()
    else:
        raw.replace(destination)  # Downloadable; browser playback may require H.264 conversion.
    return destination
