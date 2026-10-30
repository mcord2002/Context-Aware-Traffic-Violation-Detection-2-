"""Component 03: local, CPU-capable proof-of-concept.
Automatic ID recovery is deliberately NOT asserted without ground truth.
"""
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
OUTPUTS = ROOT / 'outputs'
OUTPUTS.mkdir(exist_ok=True)
YOLO_PATH = MODELS / 'yolo26n_best.pt'
RESNET_PATH = MODELS / 'resnet18_best.pt'
ACCIDENT_THRESHOLD = 0.50  # provisional threshold selected from limited validation videos
COLORS = {'car': (255, 140, 40), 'motorcycle': (220, 50, 220),
          'bus': (40, 185, 40), 'truck': (0, 215, 245)}

st.set_page_config(page_title='Component 03 | Traffic Intelligence', page_icon='🚗', layout='wide')

@st.cache_resource(show_spinner='Loading YOLO26n...')
def load_yolo():
    if not YOLO_PATH.is_file():
        raise FileNotFoundError(f'Copy your fine-tuned YOLO checkpoint to {YOLO_PATH}')
    model = YOLO(str(YOLO_PATH))
    expected = ['car', 'motorcycle', 'bus', 'truck']
    actual = [model.names[i] for i in range(4)]
    if actual != expected:
        raise ValueError(f'Unexpected class mapping: {actual}; expected {expected}')
    return model

@st.cache_resource(show_spinner='Loading ResNet18...')
def load_accident():
    if not RESNET_PATH.is_file():
        raise FileNotFoundError(f'Copy your ResNet18 checkpoint to {RESNET_PATH}')
    try:
        ckpt = torch.load(str(RESNET_PATH), map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f'ResNet checkpoint could not be read: {e}') from e
    if 'model_state_dict' not in ckpt or 'class_to_idx' not in ckpt:
        raise ValueError('Expected the saved checkpoint dictionary from your Step 8 training notebook.')
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    idx = ckpt['class_to_idx']['accident']
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    return m, idx, tf

def probability(frame, model, accident_idx, tf):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = tf(Image.fromarray(rgb)).unsqueeze(0)
    with torch.inference_mode():
        return float(torch.softmax(model(x), dim=1)[0, accident_idx].item())

def iou(a, b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1])
    x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    intersection = max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return intersection/(aa+bb-intersection) if aa+bb>intersection else 0.

def draw_label(frame, box, label, color):
    h,w=frame.shape[:2]
    x1,y1,x2,y2=(int(v) for v in box)
    x1=max(0,min(w-1,x1));x2=max(0,min(w-1,x2))
    y1=max(0,min(h-1,y1));y2=max(0,min(h-1,y2))
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    scale=max(.38,min(.63,w/1550))
    size,base=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
    tw,th=size
    left=max(0,min(x1,w-tw-10)); top=max(th+6,y1-4)
    cv2.rectangle(frame,(left,top-th-6),(left+tw+8,top+base),color,-1)
    cv2.putText(frame,label,(left+4,top-3),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),2)

def analyze(video_path, confidence, imgsz, progress, status):
    detector = load_yolo()
    accident_model, accident_idx, preprocess = load_accident()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError('OpenCV cannot decode this MP4. Convert it to H.264 MP4 and try again.')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release(); raise ValueError('Invalid source FPS')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1,round(fps/2))
    tracking = YOLO(str(YOLO_PATH))  # new instance resets tracker for each uploaded video
    frames_data=[]; rows=[]; accident_rows=[]; votes=defaultdict(lambda:defaultdict(float))
    idx=0
    status.info('Pass 1/2: detecting vehicles, tracking IDs, and classifying accident frames (CPU may take several minutes).')
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            res=tracking.track(frame, persist=True, tracker='bytetrack.yaml', conf=confidence,
                               imgsz=imgsz, device='cpu', verbose=False)[0]
            objects=[]
            if res.boxes is not None and res.boxes.id is not None:
                xyxy=res.boxes.xyxy.cpu().numpy()
                tids=res.boxes.id.int().cpu().tolist()
                clss=res.boxes.cls.int().cpu().tolist()
                confs=res.boxes.conf.cpu().tolist()
                for box, tid, cl, cf in zip(xyxy,tids,clss,confs):
                    name=res.names[int(cl)]
                    if name not in COLORS:continue
                    box=[float(v) for v in box]
                    objects.append({'frame':idx,'tracker_id':int(tid),'class':name,'conf':float(cf),'box':box})
                    votes[int(tid)][name]+=float(cf)
                    rows.append({'frame':idx,'time_sec':round(idx/fps,3),'tracker_id':int(tid),
                                 'raw_class':name,'confidence':round(float(cf),4),
                                 'x1':box[0],'y1':box[1],'x2':box[2],'y2':box[3]})
            frames_data.append(objects)
            if idx % stride == 0:
                accident_rows.append({'frame':idx,'time_sec':idx/fps,
                                      'accident_probability':probability(frame,accident_model,accident_idx,preprocess)})
            idx+=1
            if idx%10==0:progress.progress(min(0.5,0.5*idx/max(total,idx)), text=f'Pass 1: {idx}/{total or "?"} frames')
    finally:
        cap.release()
    if idx==0:raise ValueError('The uploaded file contains no decoded frames.')
    tracks=defaultdict(list)
    for objects in frames_data:
        for obj in objects:tracks[obj['tracker_id']].append(obj)
    stable_class={tid:max(classes,key=classes.get) for tid,classes in votes.items()}
    adf=pd.DataFrame(accident_rows)
    adf['smoothed_probability']=adf.accident_probability.rolling(2,min_periods=2).mean()
    scores=adf.smoothed_probability.dropna()
    peak=float(scores.max()) if not scores.empty else None
    positive=peak is not None and peak>=ACCIDENT_THRESHOLD
    estimated=float(adf.loc[scores.idxmax(),'time_sec']) if positive else None
    # Evidence of possible occlusion, not verified identity recovery.
    candidate_rows=[]; end_candidates=[]
    for tid,history in tracks.items():
        if len(history)<5:continue
        last=history[-1]; box=last['box'];end=last['frame']
        if end>=idx-2:continue
        if box[0]<25 or box[1]<25 or box[2]>width-25 or box[3]>height-25:continue
        best=0.; occluder=None
        for obs in history[-5:]:
            for other in frames_data[obs['frame']]:
                if other['tracker_id']==tid:continue
                v=iou(obs['box'],other['box'])
                if v>best:best=v;occluder=other['tracker_id']
        if best>=.1:end_candidates.append((tid,history,best,occluder))
    for old_id,oh,ov,occluder in end_candidates:
        a=oh[-1];old_box=a['box'];old_center=np.array([(old_box[0]+old_box[2])/2,(old_box[1]+old_box[3])/2])
        prev=oh[max(0,len(oh)-5)]; prev_box=prev['box']
        dt=max(1,a['frame']-prev['frame'])
        velocity=(old_center-np.array([(prev_box[0]+prev_box[2])/2,(prev_box[1]+prev_box[3])/2]))/dt
        diag=max(1.,math.hypot(old_box[2]-old_box[0],old_box[3]-old_box[1]))
        old_area=max(1.,(old_box[2]-old_box[0])*(old_box[3]-old_box[1]))
        for new_id,nh in tracks.items():
            if new_id==old_id or len(nh)<5 or stable_class[new_id]!=stable_class[old_id]:continue
            b=nh[0];gap=b['frame']-a['frame']
            if not 0<gap<=round(fps*2):continue
            nb=b['box']; pos=np.array([(nb[0]+nb[2])/2,(nb[1]+nb[3])/2])
            d=float(np.linalg.norm(old_center+velocity*gap-pos)/diag)
            ar=float(max(1.,(nb[2]-nb[0])*(nb[3]-nb[1]))/old_area)
            if d<=1.25 and .45<=ar<=2.2:
                candidate_rows.append({'old_track_id':old_id,'new_track_id':new_id,
                    'end_time_sec':round(a['frame']/fps,2),'new_start_sec':round(b['frame']/fps,2),
                    'gap_frames':gap,'vehicle_type':stable_class[old_id],
                    'overlap_iou':round(ov,3),'normalized_distance':round(d,3),
                    'area_ratio':round(ar,3),'status':'Unverified candidate'})
    # pass two: draw stabilized classes and accident scores onto a fresh copy
    raw=OUTPUTS/'processed_raw.mp4'; final=OUTPUTS/'processed_video.mp4'
    capture=cv2.VideoCapture(str(video_path))
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():capture.release();raise RuntimeError('Could not initialize MP4 writer')
    atimes=adf.time_sec.to_numpy(); avalues=adf.smoothed_probability.to_numpy()
    status.info('Pass 2/2: rendering the annotated video.')
    i=0
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            if i<len(frames_data):
                for obj in frames_data[i]:
                    vehicle=stable_class.get(obj['tracker_id'],obj['class'])
                    draw_label(frame,obj['box'],f'{vehicle.upper()} ID:{obj["tracker_id"]}',COLORS[vehicle])
            pidx=np.searchsorted(atimes,i/fps,side='right')-1
            val=float(avalues[pidx]) if pidx>=0 else float('nan')
            if math.isnan(val):state='ANALYZING';color=(230,230,230)
            elif val>=ACCIDENT_THRESHOLD:state='POSSIBLE ACCIDENT';color=(0,50,255)
            else:state='NO ACCIDENT DETECTED';color=(40,220,80)
            overlay=frame.copy();cv2.rectangle(overlay,(0,0),(min(width,610),min(height,115)),(16,16,16),-1)
            cv2.addWeighted(overlay,.7,frame,.3,0,frame)
            cv2.putText(frame,state,(12,33),cv2.FONT_HERSHEY_SIMPLEX,.77,color,2)
            cv2.putText(frame,'Accident score: '+('N/A' if math.isnan(val) else f'{val:.3f}'),(12,68),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2)
            cv2.putText(frame,f'Time: {i/fps:.1f}s | Tracker IDs are provisional',(12,100),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            writer.write(frame);i+=1
            if i%20==0:progress.progress(.5+.5*min(1,i/max(idx,1)), text=f'Pass 2: {i}/{idx} frames')
    finally:
        capture.release();writer.release()
    import subprocess
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(raw),'-c:v','libx264',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(final)],check=True)
    progress.progress(1., text='Completed')
    tdf=pd.DataFrame(rows)
    cdf=pd.DataFrame(candidate_rows)
    tdf.to_csv(OUTPUTS/'tracking_observations.csv',index=False)
    cdf.to_csv(OUTPUTS/'occlusion_candidates.csv',index=False)
    adf.to_csv(OUTPUTS/'accident_timeline.csv',index=False)
    summary={'video_level_status':'POSSIBLE ACCIDENT' if positive else 'NO ACCIDENT DETECTED',
             'peak_accident_score':peak,'suspected_event_sec':estimated,
             'tracker_segments':len(tracks),'candidate_pairs':len(cdf),
             'verified_recoveries':0,'processing':'offline CPU prototype'}
    (OUTPUTS/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary,adf,tdf,cdf,final

st.title('Traffic Intelligence Dashboard')
st.caption('Component 03 | Occlusion-Aware Vehicle–Violation Association and Behaviour Categorization | Research prototype')
st.info('This prototype detects vehicles and possible accidents. Track IDs can fragment; occlusion matches are unverified. It does not determine accident fault or automatically prove traffic-law violations.')
with st.sidebar:
    st.header('Model configuration')
    st.write('Detector: fine-tuned YOLO26n')
    st.write('Accident classifier: fine-tuned ResNet18')
    st.write('Tracker: ByteTrack')
    conf=st.slider('Detection confidence', .10,.70,.20,.05)
    imgsz=st.select_slider('Inference image size',options=[640,800,960],value=960)
    st.caption('On CPU, 960-pixel tracking may be slow. Use 640 for a quick demo.')
    st.write('Model checkpoints')
    st.write('YOLO:', '✅ Found' if YOLO_PATH.exists() else '❌ Missing')
    st.write('ResNet18:', '✅ Found' if RESNET_PATH.exists() else '❌ Missing')

uploaded=st.file_uploader('Upload a short MP4 traffic video',type=['mp4'])
if uploaded is not None:
    st.video(uploaded)
    st.caption('For a live presentation, start with a short 10–20 second clip.')
    if st.button('Analyze video',type='primary',disabled=not (YOLO_PATH.exists() and RESNET_PATH.exists())):
        with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as f:
            f.write(uploaded.getbuffer()); temporary=Path(f.name)
        bar=st.progress(0,text='Starting analysis...')
        status=st.empty()
        try:
            with st.spinner('Running local models. This may take several minutes on CPU...'):
                result=analyze(temporary,conf,imgsz,bar,status)
            st.session_state['result']=result
            st.success('Analysis completed.')
        except Exception as exc:
            st.error(f'Processing failed: {exc}')
        finally:
            temporary.unlink(missing_ok=True)

if 'result' in st.session_state:
    summary,adf,tdf,cdf,video_path=st.session_state['result']
    a,b,c,d=st.columns(4)
    a.metric('Video result',summary['video_level_status'])
    b.metric('Peak accident score',f"{summary['peak_accident_score']:.3f}" if summary['peak_accident_score'] is not None else 'N/A')
    c.metric('Tracker segments',summary['tracker_segments'])
    d.metric('Occlusion candidates',summary['candidate_pairs'])
    if summary['suspected_event_sec'] is not None:
        st.write(f"Suspected accident evidence near **{summary['suspected_event_sec']:.1f} s** (not necessarily collision onset).")
    st.subheader('Processed traffic video')
    st.video(str(video_path))
    tabs=st.tabs(['Accident timeline','Vehicle tracking','Occlusion candidates','Violation association'])
    with tabs[0]:
        st.line_chart(adf.set_index('time_sec')[['accident_probability','smoothed_probability']])
        st.dataframe(adf,use_container_width=True)
    with tabs[1]:
        st.caption('Counts represent tracker segments/observations, not necessarily unique physical vehicles.')
        st.dataframe(tdf,use_container_width=True)
    with tabs[2]:
        st.caption('Possible identity links only. No recovered IDs are automatically accepted or represented as verified.')
        if cdf.empty:st.write('No candidates passed the conservative geometric/overlap filters in this clip.')
        else:st.dataframe(cdf,use_container_width=True)
    with tabs[3]:
        st.write('No automatically verified traffic-law violation records are available in this standalone Component 03 prototype.')
        st.caption('Violation-ID association will require verified stable vehicle identities and inputs from the relevant violation component. Do not interpret accidents as violations.')
    st.subheader('Download demonstration results')
    for label,path,mime in [
       ('Processed MP4',video_path,'video/mp4'),
       ('Vehicle tracking CSV',OUTPUTS/'tracking_observations.csv','text/csv'),
       ('Accident timeline CSV',OUTPUTS/'accident_timeline.csv','text/csv'),
       ('Occlusion candidates CSV',OUTPUTS/'occlusion_candidates.csv','text/csv')]:
        st.download_button(label,path.read_bytes(),file_name=path.name,mime=mime)
else:
    st.caption('Upload an MP4 and click Analyze video. Your previous Colab model training is not repeated.')
