from pathlib import Path
import tempfile, json, hashlib
import cv2, numpy as np
import streamlit as st
from core import analyze
st.set_page_config(page_title='Component 01 | PP1',layout='wide')
st.title('Traffic violation screening')
st.caption('PP1 prototype • Wrong-way detection in one selected lane • YOLO11n + ByteTrack + temporal rules')
st.info('Use stationary-camera footage and a straight lane. Red lights, U-turn legality, helmets and other violations are not assessed.')
file=st.file_uploader('Upload traffic footage',type=['mp4','avi','mov'])
if file:
    data=file.getvalue(); digest=hashlib.sha256(data).hexdigest()
    if st.session_state.get('digest')!=digest:
        old=st.session_state.get('temp')
        if old: old.cleanup()
        st.session_state.clear(); st.session_state.digest=digest
        st.session_state.temp=tempfile.TemporaryDirectory()
    root=Path(st.session_state.temp.name); src=root/'input.mp4'; src.write_bytes(data)
    cap=cv2.VideoCapture(str(src)); ok,frame=cap.read(); cap.release()
    if not ok: st.error('Cannot decode this video. Try an MP4 file.'); st.stop()
    st.write('Select only the lane whose permitted direction you know. Coordinates are fractions of image width and height; (0,0) is top-left.')
    left,right=st.columns(2)
    with left:
        xs=st.slider('Lane left / right',0.0,1.0,(0.1,0.9),0.01)
        ys=st.slider('Lane top / bottom',0.0,1.0,(0.2,0.95),0.01)
        ax=st.slider('Arrow start X',0.0,1.0,0.5,0.01); ay=st.slider('Arrow start Y',0.0,1.0,0.8,0.01)
        bx=st.slider('Arrow end X',0.0,1.0,0.5,0.01); by=st.slider('Arrow end Y',0.0,1.0,0.3,0.01)
        verified=st.checkbox('I verified that the green arrow is the permitted travel direction in this lane.')
        fixed=st.checkbox('The camera is stationary and this lane is approximately straight.')
    cfg={'roi':[[xs[0],ys[0]],[xs[1],ys[0]],[xs[1],ys[1]],[xs[0],ys[1]]],'arrow':[[ax,ay],[bx,by]],'scene_verified':verified,'fixed_camera':fixed,'confirm_seconds':0.5,'min_displacement':0.015}
    with right:
        h,w=frame.shape[:2]; preview=frame.copy()
        cv2.rectangle(preview,(int(xs[0]*w),int(ys[0]*h)),(int(xs[1]*w),int(ys[1]*h)),(255,190,0),2)
        cv2.arrowedLine(preview,(int(ax*w),int(ay*h)),(int(bx*w),int(by*h)),(0,220,0),3)
        st.image(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB),caption='Blue boundary: assessed lane. Green arrow: allowed direction.')
    signature=json.dumps(cfg,sort_keys=True)
    if st.button('Analyze video',disabled=not(verified and fixed)):
        st.session_state.pop('result',None)
        bar=st.progress(0.0)
        try:
            with st.spinner('Detecting vehicles and checking trajectories…'):
                report,path=analyze(src,root/'results',cfg,bar.progress)
            st.session_state.result=(signature,report,path.read_bytes())
        except Exception as exc: st.error(str(exc))
    saved=st.session_state.get('result')
    if saved and saved[0]==signature:
        _,report,video=saved
        st.subheader(report['result']); st.write(f"Assessed moving tracks: {report['assessed_tracks']}")
        st.video(video)
        if report['events']: st.dataframe(report['events'])
        st.caption(report['limitations'])
        if report.get('video_warning'): st.warning(report['video_warning'])
        st.download_button('Download annotated video',video,'annotated.mp4','video/mp4')
        st.download_button('Download report',json.dumps(report,indent=2),'report.json','application/json')
        st.download_button('Save scene settings',signature,'scene_config.json','application/json')
    elif saved: st.info('Scene settings changed. Analyze again to update the result.')
