import json
import os

import torch.backends.cuda
from utils import find_optimal_coef

from args import parse_arguments
from eval.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

eval_datasets = [
    'qasc',
    'wiki_qa',
    'quartz',
    'paws',
    'story_cloze',
    'winogrande',
    'wsc'
]

print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies.json")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
    ft_accuracies_path = os.path.join(args.save, "posthoc_ft_accuracies.json")
else:
    print(f"Evaluating {args.finetuning_mode} models.")
    ft_accuracies_path = os.path.join(args.save, f"{args.finetuning_mode}_ft_accuracies.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

control_dataset = "rte"
negation_accuracies = {}

for dataset in eval_datasets:
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned.pt"
        task_vector = -LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    elif args.finetuning_mode == "standard":
        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
        task_vector = -NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}/{args.finetuning_mode}_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/{args.finetuning_mode}_finetuned.pt"
        task_vector = -NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    # We use the validation set to choose the optimal coefficient.
    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    val_metrics = evaluate_task_vector(
        'validation',
        task_vector,
        pretrained_checkpoint,
        args,
        posthoc_linearization=args.finetuning_mode == "posthoc",
    )

    args.control_threshold = 0.95
    optimal_coef = find_optimal_coef(
        val_metrics,
        metric=f"{dataset}:top1",
        minimize=True,
        control_metric=f"{control_dataset}:top1",
        control_metric_threshold=args.control_threshold
        * pretrained_accuracies[control_dataset + "Val"],
    )

    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    test_metrics = evaluate_task_vector_at_coef(
        'test',
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

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/negations.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_negations.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_negations.json"
else:
    save_file = f"{args.save}/{args.finetuning_mode}_negations.json"

if os.path.exists(save_file):
    with open(save_file) as f:
        old_accuracies = json.load(f)
    old_accuracies.update(negation_accuracies)
    negation_accuracies = old_accuracies

with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)