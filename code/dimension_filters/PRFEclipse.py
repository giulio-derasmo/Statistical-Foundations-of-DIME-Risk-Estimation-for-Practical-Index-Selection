import numpy as np
import pandas as pd
from .AbstractFilter import AbstractFilter


class PRFEclipse(AbstractFilter):
    """
    add stuff
    """
    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]
        self.hyperparams = kwargs["hyperparams"]

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)
        dlist = self.run[(self.run.query_id == query.query_id)].doc_id.to_list()

        ### Relevant Feedback
        npos = self.hyperparams['kpos'] 
        posemb = np.mean(self.docs_encoder.get_encoding(dlist[:npos]), axis=0)  
        relevant_feedback = np.multiply(qemb, posemb)
        ### Irrelevant Feedback
        nneg = self.hyperparams['kneg'] 
        negemb = np.mean(self.docs_encoder.get_encoding(dlist[-nneg:]), axis=0) 
        irrelevant_feedback = np.multiply(qemb, negemb)
        
        ### Dimension Importance Score
        itx_vec =  self.hyperparams['alpha']*relevant_feedback - self.hyperparams['beta']*irrelevant_feedback
        return itx_vec