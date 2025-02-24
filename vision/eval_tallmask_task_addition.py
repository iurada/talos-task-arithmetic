import json
import os

import torch.backends.cuda
from utils import find_optimal_coef

from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import LinearizedTaskVector, NonLinearTaskVector
import copy

args = parse_arguments()

lam = 0.2

print("*" * 100)
print("Evaluating TALL Mask / Consensus.")
ft_accuracies_path = os.path.join(args.save, f"tallmask_ft_accuracies.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    accs = json.load(f)
    new_accs = {}
    for k in accs:
        if k.split('_')[-1] == f'{lam}':
            new_accs[k.split('_')[0]] = accs[k]
    args.finetuning_accuracies = new_accs
    print(args.finetuning_accuracies)

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

eval_datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

task_vectors = []
masks = []

# Construct TALL Mask
for dataset in eval_datasets:
    multitask_vector = []
    for task_dataset in eval_datasets:
        pretrained_checkpoint = f"{args.save}/{task_dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{task_dataset}Val/finetuned.pt"
        multitask_vector.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))
    multitask_vector = sum(multitask_vector)

    pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
    task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    with torch.no_grad():
        tall_mask = {}
        for key in task_vector.vector:
            msk = torch.ones_like(task_vector.vector[key])
            msk.mul_(torch.where(task_vector.vector[key].abs() > ((multitask_vector.vector[key] - task_vector.vector[key]) * lam), 1.0, 0.0))
            tall_mask[key] = msk
    
    task_vectors.append(task_vector)
    masks.append(tall_mask)

# Consensus
K = 2
consensus_mask = {}
with torch.no_grad():
    for tall_mask in masks:
        for key in tall_mask:
            if key not in consensus_mask:
                consensus_mask[key] = tall_mask[key].clone().detach()
            else:
                consensus_mask[key] += tall_mask[key].clone().detach()
    for key in consensus_mask:
        consensus_mask[key] = torch.where(consensus_mask[key] >= K, 1.0, 0.0)
    
    for task_vec in task_vectors:
        for key in task_vec.vector:
            task_vec.vector[key].mul_(consensus_mask[key]).mul_(tall_mask[key])

task_vector = sum(task_vectors)

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

# We use the validation set to choose the optimal coefficient.
val_metrics = evaluate_task_vector(
    task_vector,
    pretrained_checkpoint,
    args,
    posthoc_linearization=args.finetuning_mode == "posthoc",
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)

# Evaluate on the test set with the optimal coefficient.
args.eval_datasets = [dataset for dataset in eval_datasets]
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
    posthoc_linearization=args.finetuning_mode == "posthoc",
)

print("=" * 100)
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
additive_accuracies = {"test": test_metrics, "val": val_metrics}

save_file = f"{args.save}/tallmask_{lam}_additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)