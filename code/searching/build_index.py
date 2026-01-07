import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append(".")

import faiss
from memmap_interface import MemmapCorpusEncoding
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", default="deeplearning19")
    parser.add_argument("-p", "--data_dir", default="data")
    parser.add_argument("-r", "--model_name", default="contriever")
    args = parser.parse_args()

    print('load corpus')
    corpus = "msmarco-passage" if args.collection in ["deeplearning19", "deeplearning20"] else "robust04"
    corpus = corpus if args.collection in ["deeplearning19", "deeplearning20", "robust04"] else "antique"
    
    # load memmap for the corpus
    corpora_memmapsdir = f"{args.data_dir}/memmaps/{args.model_name}/corpora/{corpus}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/corpus.dat",
                                        f"{corpora_memmapsdir}/corpus_mapping.csv")
    d, d_model = docs_encoder.get_shape()
    index = faiss.IndexFlatIP(d_model)

    # Add to index
    data = docs_encoder.get_data()
    print('add corpus to FAISS index')
    for s in range(0, d, 1024):
        e = min(s + 1024, d)
        keys = data[s:e]
        index.add(keys)
        print(s)

    faiss_path = f"{args.data_dir}/vectordb/{args.model_name}/corpora/{corpus}/index_db.faiss"
    print('save to: ', faiss_path)
    faiss.write_index(index, faiss_path)
