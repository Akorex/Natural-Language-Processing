This directory contains files, and notebooks pertaining to the text-generation task. 

Aims to provide a low-level understanding of the text generation task with transformers. Transformers are a ubiquitous deep learning architecture based on the attention mechanism and are mainly used for sequence-to-sequence tasks such as translation and text generation. We can also use the transformer architecture for sentiment analysis tasks, but this would require using only the encoder, as in the case of BERT.
A very helpful resource for a deep dive into the low-level understanding of the transformer architecture, its implementation, and its use for the translation task is provided in the Translation subdirectory.


Here's the objective for the task at hand:

1. Use the [HuggingFace Transformers library](https://github.com/Akorex/Natural-Language-Processing/blob/main/Text%20Generation/introduction-to-text-generation.ipynb) for a high-level understanding of what the task requires.
2. Finetune a HuggingFace Transformer model (like GPT-2) for the generation of some task
3. Build a model from scratch using [the HuggingFace Transformers](https://github.com/Akorex/Natural-Language-Processing/blob/main/Text%20Generation/text%20generation%20from%20scratch%20(huggingface).ipynb) and KerasNLP libraries
4. Use the Lower-level TensorFlow API to build a model which can be used for the task
5. Deploy the model into the ML Model Playground Repository
