This repository contains different tasks on Natural Language Processing like Sentiment Analysis, Question Answering and Named-entity recognition on several datasets from Kaggle and other sources



## Amazon Review Sentiment Analysis:

This project was based on the [Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) dataset on Kaggle.
Several models were built from scratch in order for this achieving 91% accuracy on the test dataset. 

1. Tokenization was done with the keras Tokenizer and the models were trained on 400,000 text samples from the dataset.
2. Tokenizer used was saved in a .pickle file for future used.
3. Models used are Conv1D and the LSTM models with different strengths of regularization. 
4. Glove pretrained embeddings was used to train the model and achieved a 92% accuracy on the test dataset (contains 40,000 text samples)

In the second notebook 'amazon review sentiment analysis with transformers', a transformer block was built from scratch using keras on the dataset. Decent performance was reached (~89% accuracy on the test) but not as good as the previous models built. 

A final approach on the project would be to finetune BERT (Bidirectional Encoder Representations from Transformers) on the dataset using HuggingFace ecosystem and evaluating the performance to compare with the models built from scratch.

## IMDb Review Classification
