"""Batch ingestion and sequential extraction, with per-video error reporting."""
from pathlib import Path
import hashlib

def save_uploads(uploads, folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    saved=[]; seen=set()
    for upload in uploads:
        data=upload.getvalue(); digest=hashlib.sha256(data).hexdigest()[:16]
        if digest in seen: continue
        seen.add(digest)
        path=folder/f'{digest}{Path(upload.name).suffix.lower()}'
        if not path.exists(): path.write_bytes(data)
        saved.append({'name':upload.name,'path':path,'video_id':digest})
    return saved

def extract_batch(videos, runs, extractor, weights, max_seconds, progress=None):
    results=[]
    for index,video in enumerate(videos):
        out=Path(runs)/video['video_id']
        if (out/'features.csv').exists() and (out/'metadata.json').exists():
            status='Cached — reused existing extraction'; error=''
        else:
            try:
                extractor(video['path'],runs,weights=weights,max_seconds=max_seconds)
                status='Extracted'; error=''
            except Exception as exc:
                status='Failed'; error=str(exc)
        results.append({'video':video['name'],'video_id':video['video_id'],'status':status,'error':error})
        if progress: progress((index+1)/len(videos))
    return results
