"""CLI alternative to Streamlit: python train.py data.csv --test-groups recording3"""
import argparse
import pandas as pd
from risk_core import train_model
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('csv'); parser.add_argument('--test-groups',nargs='+',required=True)
    parser.add_argument('--output',default='artifacts')
    args=parser.parse_args()
    result=train_model(pd.read_csv(args.csv),args.output,args.test_groups)
    print(f"Macro F1: {result['macro_f1']:.3f}; balanced accuracy: {result['balanced_accuracy']:.3f}")
