import json
import os

import torch.backends.cuda
from utils import find_optimal_coef

from args import parse_arguments
from eval.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

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
task_vectors = []

for dataset in eval_datasets:
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned.pt"
        task_vectors.append(
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    elif args.finetuning_mode == "standard":
        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}/{args.finetuning_mode}_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/{args.finetuning_mode}_finetuned.pt"
        task_vectors.append(
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            if 'linear' in args.finetuning_mode else
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

task_vector = sum(task_vectors)

args.eval_datasets = [dataset for dataset in eval_datasets]
args.control_dataset = None

# We use the validation set to choose the optimal coefficient.
val_metrics = evaluate_task_vector(
    'validation',
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
    'test',
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

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/additions.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_additions.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_additions.json"
else:
    save_file = f"{args.save}/{args.finetuning_mode}_additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)
