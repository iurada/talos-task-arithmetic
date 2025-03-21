# Efficient Model Editing with Task-Localized Sparse Fine-tuning [ICLR 2025]
Official code of our work "Efficient Model Editing with Task-Localized Sparse Fine-tuning" accepted at ICLR 2025.

Link to our Paper: https://openreview.net/forum?id=TDyE2iuvyc

## Introduction
<i>Task arithmetic has emerged as a promising approach for editing models by representing task-specific knowledge as composable task vectors. However, existing methods rely on network linearization to derive task vectors, leading to computational bottlenecks during training and inference. Moreover, linearization alone does not ensure weight disentanglement, the key property that enables conflict-free composition of task vectors. To address this, we propose TaLoS which allows to build sparse task vectors with minimal interference without requiring explicit linearization and sharing information across tasks. We find that pre-trained models contain a subset of parameters with consistently low gradient sensitivity across tasks, and that sparsely updating only these parameters allows for promoting weight disentanglement during fine-tuning. Our experiments prove that TaLoS improves training and inference efficiency while outperforming current methods in task addition and negation. By enabling modular parameter editing, our approach fosters practical deployment of adaptable foundation models in real-world applications.</i>

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

# Citation
```
@inproceedings{iurada2025efficient,
  author={Iurada, Leonardo and Ciccone, Marco and Tommasi, Tatiana},
  booktitle={ICLR}, 
  title={Efficient Model Editing with Task-Localized Sparse Fine-tuning}, 
  year={2025}
}
```
