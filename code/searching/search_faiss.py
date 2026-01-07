import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


import sys
sys.path.append(".")
sys.path.insert(0,'/hdd4/giuder/progetti/DIME')

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from memmap_interface import MemmapCorpusEncoding, MemmapQueriesEncoding
import argparse

def search_faiss(args, k=1000):

    corpus = "msmarco-passage" if args.collection in ["deeplearning19", "deeplearning20", "deeplearninghd"] else "robust04"
    corpus = corpus if args.collection in ["deeplearning19", "deeplearning20", "deeplearninghd", "robust04"] else "antique"
    # read the queries
    query_reader_params = {'sep': "\t", 'names': ["query_id", "text"], 'header': None, 'dtype': {"query_id": str}}
    queries = pd.read_csv(f"/hdd4/giuder/progetti/Eclipse/{args.data_dir}/queries/{args.collection}/{args.queries_tsv_filename}.tsv", **query_reader_params)

    # read qrels
    qrels_reader_params = {'sep': " ", 'names': ["query_id", "doc_id", "relevance"], "usecols": [0,2,3],
                            'header': None, "dtype": {"query_id": str, "doc_id": str}}
    qrels = pd.read_csv(f"/hdd4/giuder/progetti/Eclipse/{args.data_dir}/qrels/{args.collection}/qrels.txt", **qrels_reader_params)

    # keep only queries with relevant docs
    queries = queries.loc[queries.query_id.isin(qrels.query_id)]

    
    memmapsdir = f"/hdd4/giuder/progetti/Eclipse/{args.data_dir}/memmaps/{args.model_name}/{args.collection}"
    qrys_encoder = MemmapQueriesEncoding(f"{memmapsdir}/{args.queries_tsv_filename}.dat",
                                        f"{memmapsdir}/{args.queries_mapping_filename}.tsv")

    # load faiss index
    faiss_path = f"/hdd4/giuder/progetti/Eclipse/{args.data_dir}/vectordb/{args.model_name}/corpora/{corpus}/"
    index_name = "index_db.faiss"
    index = faiss.read_index(faiss_path + index_name)
    
    # mapper
    corpora_memmapsdir = f"/hdd4/giuder/progetti/Eclipse/{args.data_dir}/memmaps/{args.model_name}/corpora/{corpus}"
    mapping = pd.read_csv(f"{corpora_memmapsdir}/corpus_mapping.csv", dtype={'did': str})
    mapper = mapping.set_index('offset').did.to_list()
    
    
    qembs = qrys_encoder.get_encoding(queries.query_id.to_list())
    ip, idx = index.search(qembs, k)
    nqueries = len(ip)
    out = []
    for i in range(nqueries):
        run = pd.DataFrame(list(zip([queries.iloc[i]['query_id']] * len(ip[i]), idx[i], ip[i])), columns=["query_id", "did", "score"])
        run.sort_values("score", ascending=False, inplace=True)
        run['did'] = run['did'].apply(lambda x: mapper[x])
        run['rank'] = np.arange(len(ip[i]))
        out.append(run)
    out = pd.concat(out)
    out["Q0"] = "Q0"
    out["run"] = args.model_name.replace('_', '-')
    out = out[["query_id", "Q0", "did", "rank", "score", "run"]]

    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", default="deeplearning19")
    parser.add_argument("-r", "--model_name", default="contriever")
    parser.add_argument("-p", "--data_dir", default="data")
    parser.add_argument("-q", "--queries_tsv_filename", default="queries")
    parser.add_argument("-m", "--queries_mapping_filename", default="queries_mapping")

    args = parser.parse_args()

    out = search_faiss(args, k=1000)
    out.to_csv(f"/hdd4/giuder/progetti/Eclipse/{args.data_dir}/runs/{args.collection}/{args.model_name}_{args.queries_tsv_filename}.tsv", header=None, index=None, sep="\t")