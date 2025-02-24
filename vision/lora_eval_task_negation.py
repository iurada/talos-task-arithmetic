import json
import os

import torch.backends.cuda
from utils import find_optimal_coef

from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import LinearizedTaskVector, NonLinearTaskVector, LinLoRATaskVector

args = parse_arguments()


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

print("*" * 100)
print("Evaluating linear LoRA models.")
ft_accuracies_path = os.path.join(args.save, "linlora_ft_accuracies.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

control_dataset = "ImageNet"
negation_accuracies = {}

for dataset in eval_datasets:
    pretrained_checkpoint = f"{args.save}/{dataset}Val/linlora_zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/linlora_finetuned.pt"
    task_vector = -LinLoRATaskVector(pretrained_checkpoint, finetuned_checkpoint)

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

save_file = f"{args.save}/linlora_negations.json"
with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)