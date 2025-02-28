# Efficient Model Editing with Task-Localized Sparse Fine-tuning [ICLR 2025]
Official code of our work "Efficient Model Editing with Task-Localized Sparse Fine-tuning" accepted at ICLR 2025.

## Introduction
<i>Pre-trained models are stepping stones for modern machine learning systems, but how to efficiently extract, reuse, and steer their knowledge for new tasks is an area of research with still several open questions. State-of-the-art task arithmetic solutions are strongly tied to model linearization which leads to computational bottlenecks during training and inference, and potentially neglect essential task dependencies. In this work, we focus on the fine-tuning stage that defines task vectors and propose TaLoS, a new approach based on sparse fine-tuning that strategically updates only parameters expected to provide functional task localization. This efficiently yields weight-disentangled models without the need for explicit linearization. We present a thorough experimental analysis showing how our approach significantly improves in training and inference efficiency while outperforming state-of-the-art approaches in task addition and task negation. Our work offers a principled solution to pre-trained model editing and paves the way to more cost-effective and scalable machine learning systems for real-world applications.</i>

# Setting up the environment
### Requirements
Make sure to have a CUDA capable device, running at learst CUDA 11.7. Throughout our experiments we used Python version 3.10.9

### General Dependencies
To install all the required dependencies go to the root folder of this project and run:
```bash
pip install -r requirements.txt
```

### Datasets
To download and prepare the datasets for the vision experiments, please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1).

At this point you should be able to run the provided code.

## Running The Experiments
Please refer to the `args.py` file in both `vision/` and `nlp/` folders for the full list of command line arguments available. You can find in the `launch_scripts/` folder an example script used to run the experiments.

## Acknowledgement
Our code is developed starting from the one provided by the authors of ["Editing Models with Task Arithmetic"](https://arxiv.org/abs/2212.04089). Their original code repository can be found at: https://github.com/mlfoundations/task_vectors
