This directory contains files and notebooks pertaining to the Translation task.

Aims to provide a low-level understanding of the Transformer architecture and then used for the Translation task. The notebook follows the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper and the official TensorFlow tutorial for the implementation of the architecture and extended to the task of translating between language pairs.

Here's the objectives for the task at hand:

1. Write the lower-level tensorflow code for training a transformer model for the task, export the model for inference.

2. Rewrite the code using only Keras Functional API for easy reusable modules and for scaling up training

3. Use a HuggingFace Transformer model to achieve higher performance on the task

4. Use the lang-detect API to automatically detect input language and then switch around the models to translate to the appropriate language. 

5. Deploy the final approaches to the ML Model Playground repository. 