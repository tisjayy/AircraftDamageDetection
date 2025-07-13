# Aircraft Damage Detection with VGG16 and BLIP

This project detects aircraft surface damage and classifies it as either a "dent" or a "crack" using a pre-trained VGG16 model. The model was fine-tuned on a small custom dataset of aircraft damage images. It achieved around 78% test accuracy after 5 training epochs.

In addition to classification, the project uses a pre-trained BLIP model (from HuggingFace Transformers) to generate image captions and summaries. This helps describe the damage in natural language, which could support inspection teams or documentation systems.

The full workflow includes loading and preprocessing the dataset, training a custom classifier on top of VGG16, evaluating its performance, and generating captions using BLIP. The code is written in Python using TensorFlow/Keras and PyTorch for the BLIP model.
