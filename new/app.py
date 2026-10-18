"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import hashlib
import json
import time
import pandas as pd
import streamlit as st
from risk_core import STATES, FEATURES, train_model, load_model, predict_windows
from video_pipeline import extract_video, preview_pair, render_predictions
from batch import save_uploads, extract_batch

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; RUNS=DATA/'runs'; ART=ROOT/'artifacts'
for folder in [DATA/'videos',RUNS,ART]: folder.mkdir(parents=True,exist_ok=True)
LABELS=DATA/'annotations.csv'
st.set_page_config(page_title='Component 04 | Traffic Risk',layout='wide')
st.title('Component 04 · Vehicle-Pair Risk')
st.caption('PP1 pilot · five-second image-space interaction windows · one Random Forest model')
page=st.sidebar.radio('Workflow',['1 · Prepare & annotate','2 · Train & evaluate','3 · Video demo'])
st.sidebar.info('Use original source-recording groups. Clips from the same recording must share one group. Four-state training requires all four states in both train and test.')
weights=st.sidebar.text_input('Detector weights','yolov8n.pt',help='Default COCO weights download on first use. Or enter a local detector .pt path.')
seconds=st.sidebar.slider('Maximum seconds per extraction',10,300,60,10)

def labels():
    if LABELS.exists(): return pd.read_csv(LABELS,keep_default_na=False)
    return pd.DataFrame(columns=['window_id','video_id','source_group','label','evidence','reviewer'])

def dataset():
    tables=[pd.read_csv(p) for p in sorted(RUNS.glob('*/features.csv'))]
    feats=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    ann=labels()
    if feats.empty or ann.empty: return pd.DataFrame()
    return feats.merge(ann,on=['window_id','video_id'],validate='one_to_one')

def upload_video(key):
    file=st.file_uploader('Choose road footage (MP4 recommended)',type=['mp4','avi','mov','mkv'],key=key)
    if file:
        content=file.getvalue(); digest=hashlib.sha256(content).hexdigest()[:16]
        path=DATA/'videos'/f'{digest}{Path(file.name).suffix.lower()}'
        if not path.exists(): path.write_bytes(content)
        return path,digest
    return None,None

def run_extraction(path,digest,key):
    out=RUNS/digest
    if st.button('Detect, track & extract pair windows',key=key):
        with st.spinner('Tracking footage. CPU processing can take several minutes.'):
            bar=st.progress(0.)
            try:
                _,_,meta,_=extract_video(path,RUNS,weights=weights,max_seconds=seconds,callback=bar.progress)
                st.success(f"Extracted {meta['processed_seconds']:.1f} seconds in {meta['elapsed_seconds']:.1f} seconds.")
            except Exception as exc:
                st.error(f'Extraction failed: {exc}'); return None
    if (out/'features.csv').exists(): return out
    return None

if page.startswith('1'):
    st.subheader('Prepare footage and review vehicle-pair interactions')
    st.write('Use several independent recordings. Accident/normal folder names are not four-state annotations. First inspect the video and target IDs, then label the state at the window end using only evidence inside that window.')
    with st.expander('Annotation guide',expanded=False):
        st.markdown('''- **Normal:** stable interaction; no clear evasive response or developing conflict.
- **Increasing Risk:** closing movement or converging paths supported by a second cue such as shrinking visible gap or changing direction.
- **Developing Conflict:** converging/close interaction plus observable braking, steering, or other avoidance evidence.
- **Critical Conflict / Collision:** clearly severe imminent interaction or visible contact. Record which is observed; this combined state does not prove collision.

Reject uncertain pairs, identity switches, unrelated overlapping boxes and heavily occluded interactions. A box overlap alone is not a collision. Review the same state consistently across videos, ideally with a second reviewer. Do not use footage after the window end to label an early-warning state.''')
    uploads=st.file_uploader('Upload one or more road videos',type=['mp4','avi','mov','mkv'],accept_multiple_files=True,key='annotation_batch')
    videos=save_uploads(uploads,DATA/'videos') if uploads else []
    path=None; digest=None
    if videos:
        st.caption(f'{len(videos)} unique videos selected. Extraction runs sequentially to limit memory use. Existing extractions are reused with their original settings.')
        if st.button('Extract ALL uploaded videos',type='primary'):
            with st.spinner('Processing the batch. Leave this page open until it finishes.'):
                bar=st.progress(0.)
                summary=extract_batch(videos,RUNS,extract_video,weights,seconds,progress=bar.progress)
                st.session_state['batch_summary']=summary
        if 'batch_summary' in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state['batch_summary']),use_container_width=True)
        counts=labels().groupby('video_id').size().to_dict()
        st.dataframe(pd.DataFrame([{'video':v['name'],'extraction_ready':(RUNS/v['video_id']/'metadata.json').exists(),
                                   'reviewed_windows':counts.get(v['video_id'],0)} for v in videos]),use_container_width=True)
        selected=st.selectbox('Video to review and annotate',range(len(videos)),format_func=lambda i:videos[i]['name'])
        path=videos[selected]['path']; digest=videos[selected]['video_id']
    if path:
        st.video(str(path))
        out=run_extraction(path,digest,'extract_annotation')
        if out:
            feats=pd.read_csv(out/'features.csv'); tracks=pd.read_csv(out/'tracks.csv')
            st.write(f'{len(feats)} qualifying pair windows. No windows is not evidence of Normal driving.')
            if not feats.empty:
                ann=labels(); existing=ann[ann.video_id==digest]
                default_group=existing.source_group.iloc[0] if not existing.empty else ''
                group=st.text_input('Original source recording group (required)',value=default_group,placeholder='camera01_original_recording01',key=f'group_{digest}')
                st.caption('Use exactly the same group for every clip from that original recording. For stronger generalization, hold out entire cameras later.')
                index=st.selectbox('Window to review',range(len(feats)),format_func=lambda i:f"{feats.iloc[i].start_s:.0f}–{feats.iloc[i].end_s:.0f}s · pair {int(feats.iloc[i].pair_a)}–{int(feats.iloc[i].pair_b)}"+(' · labelled' if feats.iloc[i].window_id in set(ann.window_id) else ''))
                row=feats.iloc[index]
                picture=preview_pair(path,tracks,row)
                if picture is not None: st.image(picture,caption='Target pair across this window. Replay the original video for braking, steering and contact evidence.')
                with st.form(f'label_{row.window_id}'):
                    state=st.selectbox('Risk state',STATES)
                    evidence=st.text_area('Evidence: describe at least two visible cues, or why the interaction is normal')
                    reviewer=st.text_input('Reviewer name/initials')
                    checked=st.checkbox('I checked both IDs remain on the same vehicles and reviewed only this time window.')
                    save=st.form_submit_button('Save reviewed annotation')
                    if save:
                        if not group.strip() or len(evidence.strip())<10 or not reviewer.strip() or not checked:
                            st.error('Enter group, evidence, reviewer and confirm the tracking check.')
                        elif not existing.empty and set(existing.source_group)!={group.strip()}:
                            st.error('This video already has a different group. Correct its CSV consistently before continuing.')
                        else:
                            ann=ann[ann.window_id!=row.window_id]
                            record={'window_id':row.window_id,'video_id':digest,'source_group':group.strip(),'label':state,'evidence':evidence.strip(),'reviewer':reviewer.strip()}
                            pd.concat([ann,pd.DataFrame([record])],ignore_index=True).to_csv(LABELS,index=False)
                            st.success('Annotation saved. Select the next window.')
    ann=labels()
    if not ann.empty:
        st.subheader('Saved annotations')
        st.dataframe(ann,use_container_width=True)
        st.download_button('Download annotations.csv',ann.to_csv(index=False),'annotations.csv','text/csv')
        df=dataset()
        st.download_button('Download labelled_features.csv for Colab',df.to_csv(index=False),'labelled_features.csv','text/csv')
        st.caption('The ZIP package does not include your videos. Back up data/ and artifacts/ after annotation and training.')

elif page.startswith('2'):
    st.subheader('Train one Random Forest and evaluate held-out footage')
    st.caption('One click trains on ALL saved reviewed annotations across extracted videos, excluding your selected held-out source groups. Raw unlabelled windows are not training examples.')
    df=dataset()
    if df.empty: st.info('Complete video extraction and annotation first.')
    else:
        st.dataframe(pd.crosstab(df.source_group,df.label).reindex(columns=STATES,fill_value=0),use_container_width=True)
        groups=sorted(df.source_group.astype(str).unique())
        held=st.multiselect('Select held-out source recordings BEFORE training',groups)
        st.warning('Every state must appear in training and test. Do not repeatedly select splits to obtain a higher score. Same-camera performance is not held-out-camera performance.')
        if st.button('Train Random Forest & evaluate',type='primary'):
            try:
                with st.spinner('Training and evaluating...'): report=train_model(df,ART,held)
                (ART/'labelled_features.csv').write_text(df.to_csv(index=False),encoding='utf-8')
                st.success('Training completed. Evaluation uses only the selected held-out recordings.')
            except Exception as exc: st.error(str(exc))
    if (ART/'evaluation.json').exists():
        report=json.loads((ART/'evaluation.json').read_text())
        a,b,c=st.columns(3)
        a.metric('Macro F1',f"{report['macro_f1']:.3f}"); b.metric('Balanced accuracy',f"{report['balanced_accuracy']:.3f}"); c.metric('Weighted kappa',f"{report['quadratic_weighted_kappa']:.3f}")
        st.caption(f"{report['train_windows']} training windows / {report['test_windows']} test windows; overlapping windows are correlated. Majority baseline macro F1: {report['majority_baseline_macro_f1']:.3f}")
        st.write('Classification report')
        st.dataframe(pd.DataFrame(report['classification_report']).T,use_container_width=True)
        st.write('Confusion matrix — rows: actual; columns: predicted')
        st.dataframe(pd.DataFrame(report['confusion_matrix'],index=STATES,columns=STATES),use_container_width=True)
        st.bar_chart(pd.read_csv(ART/'feature_importance.csv').set_index('feature'))
        with st.expander('Split provenance and limitations'): st.json(report)
        for name in ['risk_model.pkl','evaluation.json','split_manifest.csv','test_predictions.csv']:
            p=ART/name
            if p.exists(): st.download_button(f'Download {name}',p.read_bytes(),name)

else:
    st.subheader('Demonstrate risk predictions on road footage')
    model_path=ART/'risk_model.pkl'
    if not model_path.exists():
        st.info('Train the risk model first. If trained in Colab, copy its trusted risk_model.pkl into this package’s artifacts folder. No fabricated model or evaluation is bundled.')
    else:
        st.caption('Only load your own trusted pickle file. Model artifacts can execute code when loaded.')
        bundle=load_model(model_path)
        path,digest=upload_video('demo_video')
        if path:
            st.video(str(path))
            known=set(pd.read_csv(ART/'split_manifest.csv').video_id) if (ART/'split_manifest.csv').exists() else set()
            if digest in known: st.info('This exact video appears in the training/evaluation manifest. Use a new recording for an additional unseen demo; report measured test scores separately.')
            out=run_extraction(path,digest,'extract_demo')
            if out:
                features=pd.read_csv(out/'features.csv'); tracks=pd.read_csv(out/'tracks.csv')
                if features.empty: st.warning('No eligible five-second pair window. Try longer footage with stable tracks for two vehicles. This is not a Normal prediction.')
                else:
                    began=time.perf_counter(); pred=predict_windows(bundle,features); elapsed=time.perf_counter()-began
                    pred.to_csv(out/'predictions.csv',index=False)
                    st.caption(f'Risk inference: {elapsed*1000/len(pred):.2f} ms/window (excludes detection, extraction and rendering). Vote shares are uncalibrated, not collision probabilities.')
                    pairs=sorted(set(zip(pred.pair_a,pred.pair_b)))
                    a,b=st.selectbox('Inspect vehicle pair',pairs,format_func=lambda p:f'ID {p[0]} ↔ ID {p[1]}')
                    view=pred[(pred.pair_a==a)&(pred.pair_b==b)].copy()
                    view['risk_level']=view.prediction.map({s:i for i,s in enumerate(STATES)})
                    st.caption('Timeline levels: 0 Normal · 1 Increasing · 2 Developing · 3 Critical / Collision')
                    st.line_chart(view.set_index('end_s')[['risk_level']])
                    st.line_chart(view.set_index('end_s')[['separation_last']])
                    st.dataframe(view[['start_s','end_s','pair_a','pair_b','prediction','vote_share']],use_container_width=True)
                    picture=preview_pair(path,tracks,view.iloc[-1])
                    if picture is not None: st.image(picture,caption='Most recent window for this pair')
                    st.download_button('Download pair predictions CSV',pred.to_csv(index=False),'predictions.csv','text/csv')
                    if st.button('Render annotated video'):
                        try:
                            with st.spinner('Rendering video...'): render_predictions(path,tracks,pred,out/'annotated.mp4')
                        except Exception as exc: st.error(f'Rendering failed: {exc}')
                    rendered=out/'annotated.mp4'
                    if rendered.exists():
                        st.video(str(rendered))
                        st.download_button('Download annotated MP4',rendered.read_bytes(),'component04_demo.mp4','video/mp4')
                        st.caption('If browser playback fails, open the download in VLC. Install FFmpeg and render again for H.264 playback.')
