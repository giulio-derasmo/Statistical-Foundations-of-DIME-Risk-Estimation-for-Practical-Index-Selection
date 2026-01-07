# =====================
# Environment Variables
# =====================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================
# Path Setup
# =====================
import sys
if not os.getcwd().endswith("Eclipse"):
    os.chdir('../../Eclipse')
    sys.path.append("..")

# =====================
# Standard Libraries
# =====================
import itertools
from multiprocessing import Pool
from functools import partial
from copy import deepcopy
from glob import glob
from collections import Counter
import argparse
import pickle
from typing import Tuple
from ast import literal_eval

# =====================
# Scientific Libraries
# =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from scipy.special import softmax
from sklearn.gaussian_process.kernels import RBF
# =====================
# External Libraries
# =====================
import faiss
import ir_measures
from sentence_transformers import SentenceTransformer

# =====================
# Local Modules
# =====================
import Eclipse.dimension_filters as dimension_filters
#from dimension_filters.AbstractFilter import AbstractFilter
from Eclipse.memmap_interface import MemmapCorpusEncoding, MemmapQueriesEncoding

from utils import compute_variance_true, modified_masked_retrieve_and_evaluate

# =====================
# Constants
# =====================
collection2corpus = {
    "deeplearning19": "msmarco-passage",
    "deeplearning20": "msmarco-passage",
    "deeplearninghd": "msmarco-passage",
    "robust04": "robust04",
    "antique": "antique"
}

m2hf = {
    "tasb": "sentence-transformers/msmarco-distilbert-base-tas-b",
    "contriever": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "cocondenser": "sentence-transformers/msmarco-bert-co-condensor"
}



def main(args):
    
    ### ---------------- LOAD STUFF ---------------------
    print('Load FAISS index')
    faiss_path = f"{args.datadir}/vectordb/{args.model_name}/corpora/{collection2corpus[args.collection]}/"
    index_name = "index_db.faiss"
    index = faiss.read_index(faiss_path + index_name)

    # read the queries
    query_reader_params = {'sep': "\t", 'names': ["query_id", "text"], 'header': None, 'dtype': {"query_id": str}}
    queries = pd.read_csv(f"{args.datadir}/queries/{args.collection}/queries.tsv", **query_reader_params)
    # read qrels
    qrels_reader_params = {'sep': " ", 'names': ["query_id", "doc_id", "relevance"], "usecols": [0,2,3],
                            'header': None, "dtype": {"query_id": str, "doc_id": str}}
    qrels = pd.read_csv(f"{args.datadir}/qrels/{args.collection}/qrels.txt", **qrels_reader_params)
    # keep only queries with relevant docs
    queries = queries.loc[queries.query_id.isin(qrels.query_id)]

    # load memmap for the corpus
    corpora_memmapsdir = f"{args.datadir}/memmaps/{args.model_name}/corpora/{collection2corpus[args.collection]}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/corpus.dat",
                                        f"{corpora_memmapsdir}/corpus_mapping.csv")

    memmapsdir = f"{args.datadir}/memmaps/{args.model_name}/{args.collection}"
    qrys_encoder = MemmapQueriesEncoding(f"{memmapsdir}/queries.dat",
                                        f"{memmapsdir}/queries_mapping.tsv")

    mapping = pd.read_csv(f"{corpora_memmapsdir}/corpus_mapping.csv", dtype={'did': str})
    mapper = mapping.set_index('offset').did.to_list()

    q2r = pd.DataFrame({"query_id": queries.query_id.to_list(), "row": np.arange(len(queries.query_id.to_list()))})

    ### ---------------- Filter selection  ---------------------
    if args.filter_function == "GPTFilter":
        #print('Load LLM DIME')
        model = SentenceTransformer(m2hf[args.model_name])
        answers_path = f"{args.datadir}/runs/{args.collection}/chatgpt4_answer.csv"
        filtering = dimension_filters.GPTFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, model=model, answers_path=answers_path)
  
    elif args.filter_function == "OracularFilter":
        #print('Load Oracle DIME')
        filtering = dimension_filters.OracularFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder)
    
    elif args.filter_function == "TopkFilter":
        #print('Load PRF DIME')
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank", "score"], 'usecols': [0, 2, 3, 4], 'sep': "\t",
                        'dtype': {"query_id": str, "doc_id": str, "rank": int, "score": float}}
        run = pd.read_csv(f"{args.datadir}/runs/{args.collection}/{args.model_name}.tsv", **run_reader_parmas)
        kpos = 2
        filtering = dimension_filters.TopkFilter(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder, run=run, k=kpos)

    elif args.filter_function == 'SWC': 
        run_reader_parmas = {'names': ["query_id", "doc_id", "rank", "score"], 'usecols': [0, 2, 3, 4], 'sep': "\t",
                        'dtype': {"query_id": str, "doc_id": str, "rank": int, "score": float}}
        run = pd.read_csv(f"{args.datadir}/runs/{args.collection}/{args.model_name}.tsv", **run_reader_parmas)
        kpos = 10
        filtering = dimension_filters.SoftTopkFilter(docs_encoder=docs_encoder, 
                                qrys_encoder=qrys_encoder, 
                                run=run, 
                                k=kpos,
                                tau=0.1, 
                                )

    rel_dims = filtering.filter_dims(queries, explode=True)
    ## compute the variance
    series_of_numpy = rel_dims.groupby("query_id").importance.apply(np.array).loc[q2r.query_id.to_list()]
    U_q = np.array(np.stack(series_of_numpy.values), dtype=np.float32)
    var = compute_variance_true(U_q, qrys_encoder.get_encoding(queries.query_id.to_list()), q2r)

    qembs = qrys_encoder.get_encoding(queries.query_id.to_list()) 
    q2r = pd.DataFrame({"query_id": queries.query_id.to_list(), "row": np.arange(len(queries.query_id.to_list()))})

    for measure_name in ['AP', 'nDCG@10']:
        output = modified_masked_retrieve_and_evaluate(queries, qrels, qembs, mapper, q2r, 
                                                    rel_dims, var, index, measure_name)
        ndims = (rel_dims[rel_dims['importance'] > rel_dims['query_id'].map(var)].groupby('query_id').size().to_numpy() / 768 * 100).round(2)
        output.loc[:, 'ndims'] = ndims
        
        print(f'Finished test {args.collection} {args.model_name} {args.filter_function} {measure_name}: ', output.value.mean())

        save_filename = f"/hdd4/giuder/progetti/Eclipse/dime_risk_thresholding/output/results/{args.collection}_{args.model_name}_{args.filter_function}_{measure_name}.csv"
        output.to_csv(save_filename, index=False)


if __name__ == "__main__":
    
    tqdm.pandas()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", default="deeplearning19")
    parser.add_argument("-r", "--model_name", default="contriever")
    parser.add_argument("-d", "--datadir",    default="data")
    parser.add_argument("-f", "--filter_function", default="OracularFilter")

    args = parser.parse_args()

    main(args)