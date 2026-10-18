"""Batch extraction, preserving video hashes and metadata. No auto-generated labels."""
import argparse
from pathlib import Path
from video_pipeline import extract_video
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('folder'); parser.add_argument('--output',default='data/runs')
    parser.add_argument('--weights',default='yolov8n.pt'); parser.add_argument('--seconds',type=float,default=120)
    args=parser.parse_args()
    for video in sorted(Path(args.folder).rglob('*')):
        if video.suffix.lower() in ['.mp4','.avi','.mov','.mkv']:
            try:
                _,features,meta,out=extract_video(video,args.output,weights=args.weights,max_seconds=args.seconds)
                print(video.name,len(features),'pair windows',out)
            except Exception as exc: print('FAILED',video,exc)
