# SI 650: Information Retrieval

---

**University of Michigan / Spring 2022**

This repository is my implementation of assignments and projects for the [Information Retrieval](https://www.si.umich.edu/programs/courses/650#gsc.tab=0) course.

Topics covered in the course include:

- Text processing
- Inverted indexes
- Retrieval models (vector spaces and probabilistic),
- Clustering
- Topic modeling
- Deep learning
- Retrieval system design
- Web search engines.

## Assignment 1
---

Probabilistic models of retrieval and simple text analysis

✅ Q1: Probabilistic Reasoning and Bayes Rule

✅ Q2: Tokenization, SpaCy POS Tagging, TF-IDF

✅ Q3: Document Ranking and Evaluation

✅ Q4: Simple Search Engine (cosine similarity search)

## Assignment 2
---

Building a ranking function and simple search engine

✅ Q1: Re-implement 2 vector space model (VSM) ranking functions: BM25 and Pivoted Length Normalization

✅ Q2: Implement a custom scoring function that beats an untuned BM25 ranker by NDCG@5


## Assignment 3
---

Training a deep learning retrieval system (PyTerrier) with GPUs on the CORD19 test collection

✅ Q1: Index the CORD19 dataset, perform a BatchRetrieve search, and test models with an Experiment object

✅ Q2: Learning to Rank approaches with custom features and evaluation metrics

✅ Q3: Use BERT to re-rank content, a text-to-text model to perform query augmentation, and train a deep learning IR model to compare performance