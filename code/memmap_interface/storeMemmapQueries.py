import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('.') 
sys.path.insert(0,'/hdd4/giuder/progetti/DIME')

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm.autonotebook import tqdm
from time import time
import argparse
from normalize_text import normalize
import pandas as pd
from utils import load_collection


m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "glove": 'sentence-transformers/average_word_embeddings_glove.6B.300d',
        "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
        "cocondenser": 'sentence-transformers/msmarco-bert-co-condensor'}

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("-r", "--model_name")
        parser.add_argument("-p", "--data_dir", default="data")
        parser.add_argument("-c", "--collection", default="deeplearning19")
        parser.add_argument("-q", "--queries_tsv_filename", default="queries")
        parser.add_argument("-m", "--queries_mapping_filename", default="queries_mapping")
        args = parser.parse_args()

        queries = []
        qpath = f"/hdd4/giuder/progetti/Eclipse/{args.data_dir}/queries/{args.collection}/{args.queries_tsv_filename}.tsv"
        queries = pd.read_csv(qpath, sep="\t", header=None, names=["qid", "text"], dtype={"qid": str})
        queries["offset"] = np.arange(len(queries.index))
        model = SentenceTransformer(m2hf[args.model_name])

        repr = np.array(queries.text.apply(model.encode).to_list())
        fp = np.memmap(f"/hdd4/giuder/progetti/Eclipse/{args.data_dir}/memmaps/{args.model_name}/{args.collection}/{args.queries_tsv_filename}.dat", dtype='float32', mode='w+', shape=repr.shape)
        fp[:] = repr[:]
        fp.flush()

        queries[['qid', 'offset']].to_csv(f"/hdd4/giuder/progetti/Eclipse/{args.data_dir}/memmaps/{args.model_name}/{args.collection}/{args.queries_mapping_filename}.tsv", sep="\t", index=False)