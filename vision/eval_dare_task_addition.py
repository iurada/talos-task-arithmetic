import json
import os

import torch.backends.cuda
from utils import find_optimal_coef

from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

sparsity = 0.90


print("*" * 100)
print("Evaluating DARE models.")
ft_accuracies_path = os.path.join(args.save, f"dare_ft_accuracies.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    accs = json.load(f)
    new_accs = {}
    for k in accs:
        if k.split('_')[-1] == f'{sparsity}':
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

for dataset in eval_datasets:
    pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

    task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    # Drop And REscale (DARE)
    if sparsity > 0.0: # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
        with torch.no_grad():
            for key in task_vector.vector:
                if 'positional_embedding' in key or 'text_projection' in key or 'logit_scale' in key:
                    continue
                score = torch.randn_like(task_vector.vector[key])
                threshold, _ = torch.kthvalue(torch.flatten(score), k=int(sparsity * score.numel()))
                task_vector.vector[key].mul_(torch.where(score <= threshold, 0.0, 1.0 / (1.0 - sparsity)))
        
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

save_file = f"{args.save}/dare_{sparsity}_additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)