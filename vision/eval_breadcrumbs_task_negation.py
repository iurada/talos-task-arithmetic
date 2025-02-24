import json
import os

import torch.backends.cuda
from utils import find_optimal_coef

from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

sparsity = 0.9

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

print("*" * 100)
print("Evaluating Breadcrumbs models.")
ft_accuracies_path = os.path.join(args.save, f"breadcrumbs_ft_accuracies.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)
    print(args.finetuning_accuracies)

control_dataset = "ImageNet"
negation_accuracies = {}

for dataset in eval_datasets:
    pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
    task_vector = -NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

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

    # We use the validation set to choose the optimal coefficient.
    args.eval_datasets = [dataset + "Val"]
    args.control_dataset = control_dataset + "Val"
    val_metrics = evaluate_task_vector(
        task_vector,
        pretrained_checkpoint,
        args,
        posthoc_linearization=args.finetuning_mode == "posthoc",
    )

    args.control_threshold = 0.95
    optimal_coef = find_optimal_coef(
        val_metrics,
        metric=f"{dataset}Val:top1",
        minimize=True,
        control_metric=f"{control_dataset}Val:top1",
        control_metric_threshold=args.control_threshold
        * pretrained_accuracies[control_dataset + "Val"],
    )

    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        optimal_coef,
        posthoc_linearization=args.finetuning_mode == "posthoc",
    )

    print("=" * 100)
    print(f"Test accuracy: {test_metrics[f'{dataset}:top1']}")

    negation_accuracies[dataset] = {
        "test": test_metrics[f"{dataset}:top1"],
        "test_control": test_metrics[f"{control_dataset}:top1"],
        "val": val_metrics,
    }

save_file = f"{args.save}/breadcrumbs_negations.json"
with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)