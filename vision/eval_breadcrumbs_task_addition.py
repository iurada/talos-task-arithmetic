import json
import os

import torch.backends.cuda
from utils import find_optimal_coef

from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

sparsity = 0.8

print("*" * 100)
print("Evaluating Breadcrumbs models.")
ft_accuracies_path = os.path.join(args.save, f"breadcrumbs_ft_accuracies.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

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

for dataset in eval_datasets:
    pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

    task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    # Breadcrumbs
    if sparsity > 0.0: # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
        with torch.no_grad():
            top_k_keep = 0.2
            top_k_remove = 0.1
            for key in task_vector.vector:
                if any(x in key for x in ['attn', 'mlp', 'conv']):
                    tensor = task_vector.vector[key]
                    # Remove top
                    top_k_int = int(tensor.shape[-1] * top_k_remove)
                    _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
                    mask = torch.ones_like(tensor)
                    mask.scatter_(len(tensor.shape) - 1, masked_indices, 0.0)
                    tensor.mul_(mask)

                    # Keep top
                    top_k_int = int(tensor.shape[-1] * top_k_keep)
                    _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
                    mask = torch.zeros_like(tensor)
                    mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)
                    tensor.mul_(mask)
        
    task_vectors.append(task_vector)

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

save_file = f"{args.save}/breadcrumbs_additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)