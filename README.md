# Book Recommendation System

[![Project](https://img.shields.io/badge/project-book--recommender-blue.svg)](https://github.com/PouriaAsadi1/Book-Recommendation-Model)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

A scalable book recommendation system built on the Goodreads dataset. This repository contains preprocessing utilities, two recommendation model approaches (SASRec and Word2Vec-style item embeddings), evaluation scripts, and examples for training and inference.

Authors: Princely Oseji, Pouria Asadi, Tianchi Wu

---

## Table of contents

- [About](#about)
- [Features](#features)
- [Dataset](#dataset)
- [Getting started](#getting-started)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Download data](#download-data)
- [Quick start](#quick-start)
  - [Preprocess data](#preprocess-data)
  - [Train models](#train-models)
  - [Evaluate](#evaluate)
  - [Inference / Serve](#inference--serve)
- [Project structure](#project-structure)
- [Evaluation metrics](#evaluation-metrics)
- [Results (summary)](#results-summary)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Authors & contact](#authors--contact)
- [Acknowledgements & References](#acknowledgements--references)

---

## About

The goal of this project is to learn latent representations that capture complex user preferences and book characteristics by leveraging large-scale Goodreads user–book interactions and book metadata. Two complementary approaches are implemented:

- SASRec — a Transformer-based sequential recommendation model for next-item prediction.
- Word2Vec-style item embeddings — learns item-to-item similarity from interaction sequences for similarity-based retrieval and discovery.

This system focuses on reproducible offline evaluation (Precision@K, Recall@K) and is designed to be extensible to hybrid models and online evaluation.

---

## Features

- Data preprocessing pipeline for the UCSD Goodreads dataset
- SASRec implementation (sequential/temporal modeling)
- Word2Vec-style item embedding training and ANN retrieval support
- Offline evaluation scripts (Precision@K, Recall@K)
- Example CLI-style scripts for preprocessing, training, evaluation, and inference
- Configurable hyperparameters and dataset filters

---

## Dataset

We use the UCSD-collected Goodreads dataset (public, large-scale). It contains:
- User–book interactions (~4.1 GB): user_id, book_id, is_read, rating, is_reviewed (we use user_id, book_id, rating)
- Book metadata (~2.3 GB, ~2.3M books): book_id, title, authors, genres, description, average_rating

Source / reference:
- UCSD Goodreads dataset: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home

NOTE: The raw dataset is large; consider filtering or sampling for local experiments.


---

## Evaluation metrics

This project focuses on standard offline accuracy metrics:
- Precision@K — fraction of recommended items in top-K that are relevant.
- Recall@K — fraction of relevant items captured in top-K recommendations.

Other useful metrics to consider (future work): NDCG@K, MAP, diversity, novelty, calibration, and online/business metrics (CTR, retention).

---

## Acknowledgements & References

- UCSD Goodreads dataset — https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
- Kang, W.-C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation.
- Chamberlain, B.P., Rossi, E., Shiebler, D., Sedhain, S., Bronstein, M.M. (2020). Tuning Word2vec for Large Scale Recommendation Systems.
- Additional references available in repository documentation and report.

---
