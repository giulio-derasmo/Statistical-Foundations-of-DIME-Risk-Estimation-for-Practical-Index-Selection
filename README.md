# Statistical Foundations of DIME: Risk Estimation for Practical Index Selection
Repository for the code used in Statistical Foundations of DIME: Risk Estimation for Practical Index Selection published @EACL2026


## 📄 Abstract

High-dimensional dense embeddings have become central to modern Information Retrieval, but many dimensions are noisy or redundant1. Recently proposed Dimension Importance Estimation (DIME) provides query-dependent scores to identify informative components but relies on a costly grid search to select a dimensionality a priori2.This work introduces RDIME (Risk Dimension Importance Estimation), a statistically grounded criterion that directly identifies the optimal set of dimensions for each query at inference time3. By modeling the query embedding as a noisy observation of a latent information need, we derive a hard-thresholding estimator that minimizes $l_2$ risk. Experiments confirm that this approach improves retrieval effectiveness and reduces embedding size by an average of ~50% across different models and datasets4.

## 🛠️ Methodology:
RDIME Unlike top-k thresholding strategies that require fixing $k$ globally, uses a Hard Thresholding Estimator. Given a query $q$ and its DIME representation $u_q$ (derived from pseudo-relevant documents or LLM generation), we calculate a dynamic noise threshold $\hat{\epsilon}^2$. The set of retained dimensions $\hat{S}$ is defined as:

$$
\hat{S} = \\{ i \in \{1, ..., p\} \mid (u_q)_i > \hat{\epsilon}^2 \\}
$$

Where the noise level is estimated per query:

$$
\hat{\epsilon}^2 = \frac{1}{p} \sum_{i=1}^{p} (q_i^2 - (u_q)_i)
$$

The selection is performed in a query-dependent manner, allowing us to adapt the dimensionality for each query rather than relying on a single fixed value for the entire collection.

## Cite 

```{text}
@inproceedings{
d'erasmo2026statistical,
title={Statistical Foundations of {DIME}: Risk Estimation for Practical Index Selection},
author={Giulio D'Erasmo and Cesare Campagnano and Antonio Mallia and Pierpaolo Brutti and Nicola Tonellotto and Fabrizio Silvestri},
booktitle={19th Conference of the European Chapter of the Association for Computational Linguistics},
year={2026},
url={https://openreview.net/forum?id=ThNr228gyX}
}
```
