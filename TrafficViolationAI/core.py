"""PP1 prototype: fixed camera, one straight lane, manually verified direction."""
from pathlib import Path
from collections import defaultdict, deque
import json, math, uuid, subprocess

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / 'models' / 'yolo11n.pt'

def motion(history, direction, min_distance, window=0.75):
    if len(history) < 3 or history[-1][0]-history[0][0] < window*0.8:
        return None
    dx, dy = history[-1][1]-history[0][1], history[-1][2]-history[0][2]
    distance = math.hypot(dx, dy)
    if distance < min_distance:
        return None
    cosine = (dx*direction[0]+dy*direction[1])/distance
    return 'opposite' if cosine < -0.7 else ('allowed' if cosine > 0.7 else 'ambiguous')

def analyze(video, output_dir, cfg, progress=None):
    import cv2, numpy as np
    from ultralytics import YOLO
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    roi=np.asarray(cfg['roi'],dtype=float)
    a,b=np.asarray(cfg['arrow'],dtype=float)
    if roi.shape!=(4,2) or not np.isfinite(roi).all() or ((roi<0)|(roi>1)).any():
        raise ValueError('ROI must contain four normalized points.')
    if abs(cv2.contourArea(roi.astype(np.float32)))<0.001:
        raise ValueError('Selected lane area is too small.')
    if not cfg.get('scene_verified') or not cfg.get('fixed_camera'):
        raise ValueError('Verify permitted direction and stationary camera first.')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError('Cannot open video.')
    fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps<=0: cap.release(); raise ValueError('Invalid video frame rate.')
    ok, frame=cap.read()
    if not ok: cap.release(); raise ValueError('Video contains no readable frames.')
    h,w=frame.shape[:2]; scale=np.array([w,h]); polygon=(roi*scale).astype(np.int32)
    vector=(b-a)*scale; norm=np.linalg.norm(vector)
    if norm<5: cap.release(); raise ValueError('Direction arrow must have distinct endpoints.')
    direction=vector/norm
    raw=out/'annotated_raw.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened(): cap.release(); raise RuntimeError('Video writer failed.')
    histories=defaultdict(deque); last={}; streak={}; events={}; assessed=set()
    run_id=uuid.uuid4().hex[:10]; i=0; window=0.75
    # A fresh model/tracker for every clip prevents track IDs leaking between uploads.
    try:
        model=YOLO(cfg.get('weights',DEFAULT_WEIGHTS))
        while ok:
            t=i/fps
            result=model.track(frame,persist=True,tracker='bytetrack.yaml',classes=[2,3,5,7],conf=0.25,verbose=False)[0]
            canvas=frame.copy()
            cv2.polylines(canvas,[polygon],True,(255,190,0),2)
            cv2.arrowedLine(canvas,tuple((a*scale).astype(int)),tuple((b*scale).astype(int)),(0,220,0),3)
            boxes=result.boxes
            if boxes is not None and boxes.id is not None:
                for box,tid in zip(boxes.xyxy.cpu().numpy(),boxes.id.int().cpu().tolist()):
                    x1,y1,x2,y2=box; p=((x1+x2)/2,y2)
                    if cv2.pointPolygonTest(polygon,p,False)<0:
                        histories.pop(tid,None); streak.pop(tid,None); last.pop(tid,None); continue
                    if tid in last and t-last[tid]>0.25:
                        histories[tid].clear(); streak.pop(tid,None)
                    last[tid]=t; hist=histories[tid]; hist.append((t,*p))
                    while len(hist)>1 and t-hist[0][0]>window: hist.popleft()
                    state=motion(hist,direction,cfg.get('min_displacement',0.015)*math.hypot(w,h),window)
                    if state in ('allowed','opposite'): assessed.add(tid)
                    if state=='opposite':
                        streak.setdefault(tid,t)
                        if t-streak[tid]>=cfg.get('confirm_seconds',0.5):
                            if tid not in events:
                                events[tid]={'event_id':f'{run_id}-WW-{tid}','track_id':tid,'type':'wrong_way','start_seconds':round(hist[0][0],3),'confirmed_seconds':round(t,3),'end_seconds':round(t,3)}
                            events[tid]['end_seconds']=round(t,3)
                    else: streak.pop(tid,None)
                    color=(0,0,255) if tid in events else (0,220,0)
                    cv2.rectangle(canvas,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
                    cv2.putText(canvas,f'ID {tid}: '+('WRONG WAY' if tid in events else (state or 'observing')),(int(x1),max(22,int(y1)-7)),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)
            writer.write(canvas); i+=1
            if progress and i%10==0: progress(min(i/max(total,1),0.99))
            ok,frame=cap.read()
    finally:
        cap.release(); writer.release()
    incomplete=total>0 and i<total-2
    verdict='Violation detected' if events else ('Unable to determine' if not assessed or incomplete else 'No wrong-way violation detected')
    report={'result':verdict,'scope':'Wrong-way only; selected lane; manually configured direction','events':list(events.values()),'assessed_tracks':len(assessed),'processed_frames':i,'reported_frames':total,'possible_decode_truncation':incomplete,'config':cfg,'limitations':'Pretrained detector plus motion rules, not trained on the manual CSV. Other violations are not assessed. Tracking errors and missed vehicles remain possible.'}
    target=out/'annotated.mp4'
    try:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(raw),'-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True,capture_output=True)
        raw.unlink()
    except (OSError,subprocess.CalledProcessError):
        target=raw; report['video_warning']='H264 conversion unavailable; download raw MP4 if browser playback fails.'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    if progress: progress(1.0)
    return report,target
