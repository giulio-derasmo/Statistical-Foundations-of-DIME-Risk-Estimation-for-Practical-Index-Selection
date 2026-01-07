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



def compute_measure(run: pd.DataFrame, 
                    qrels: pd.DataFrame, 
                    measure_name: str) -> pd.DataFrame:
    '''
    Computes an IR (Information Retrieval) evaluation metric given a run and relevance judgments.

    Inputs:
    - "run": A DataFrame with at least columns query_id, doc_id, and score, representing the ranked results for each query.
    - "qrels": A DataFrame containing the ground-truth relevance judgments (query-document relevance pairs).
    - "measure_name": A string indicating the name of the IR evaluation measure to compute (e.g., "nDCG@10" or "MAP").
    
    Outputs: 
    - out (pd.DataFrame): A DataFrame with evaluation results. 
    '''
    measure = [ir_measures.parse_measure(measure_name)]
    out = pd.DataFrame(ir_measures.iter_calc(measure, qrels, run))
    out['measure'] = out['measure'].astype(str)
    return out


def save_pickle(data, filename):
    with open(f'{filename}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(f'{filename}.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data


def modified_masked_retrieve_and_evaluate(
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    qembs: np.ndarray,
    mapper: dict,
    q2r: pd.DataFrame,
    dim_importance: pd.DataFrame,
    var: float,
    index,
    measure: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    

    ## select only the dimensions based on K = { theta^2 > eps^2 }
    selected_dims = dim_importance.loc[dim_importance['importance'] > dim_importance['query_id'].map(var)][["query_id", "dim"]]
    ## create the mask such that q_i = 0 if i is not in top-ndims
    rows = np.array(selected_dims[["query_id"]].merge(q2r)["row"])
    cols = np.array(selected_dims["dim"])
    mask = np.zeros_like(qembs)
    mask[rows, cols] = 1

    ## apply the mask
    enc_queries = np.where(mask, qembs, 0)

    ## search
    ip, idx = index.search(enc_queries, 1000)
    ## run   
    nqueries = len(ip)
    run = []
    for i in range(nqueries):
        local_run = pd.DataFrame({"query_id": queries.iloc[i]["query_id"], "doc_id": idx[i], "score": ip[i]})
        local_run.sort_values("score", ascending=False, inplace=True)
        local_run['doc_id'] = local_run['doc_id'].apply(lambda x: mapper[x])
        run.append(local_run)

    run = pd.concat(run)
    res = compute_measure(run, qrels, measure)
    return res


def compute_variance_true(U_q, qembs, q2r = None):
    """
    Computes the variance of the importance scores for each query in U_q.
    Args:
    - U_q: A numpy array of shape (n_queries, n_dimensions) containing the importance scores for each query.
    - qembs: A numpy array of shape (n_queries, embedding_dim) containing the embeddings of the queries.
    """ 
    var =  np.mean(qembs**2 - U_q, axis=1)
    if q2r is not None: 
        var = pd.Series(var, index=q2r.query_id.to_list())
        var = var.loc[q2r.query_id.to_list()]
    return var