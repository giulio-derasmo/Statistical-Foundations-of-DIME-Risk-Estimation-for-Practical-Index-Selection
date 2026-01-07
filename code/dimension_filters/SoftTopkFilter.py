import numpy as np
from .AbstractFilter import AbstractFilter
from scipy.special import softmax

def relevance_score_kernel(q, D, tau = 1.): 
    r = np.einsum('p, Mp -> M', q, D)
    return softmax(r / tau)

class SoftTopkFilter(AbstractFilter):
    """
    """

    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]
        self.k = kwargs["k"]
        self.tau = kwargs["tau"]

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)
        dlist = self.run[(self.run.query_id == query.query_id) & (self.run["rank"] < self.k)].doc_id.to_list()
        dembs = self.docs_encoder.get_encoding(dlist)

        weights = relevance_score_kernel(qemb, dembs, tau=self.tau)
        demb = np.einsum('M, Mp -> p', weights, dembs)
        itx_vec = np.multiply(qemb, demb)

        return itx_vec