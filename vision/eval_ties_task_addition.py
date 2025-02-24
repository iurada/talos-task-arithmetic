import json
import os

import torch.backends.cuda
from utils import find_optimal_coef

from args import parse_arguments
from eval import evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import LinearizedTaskVector, NonLinearTaskVector
import copy

args = parse_arguments()

sparsity = 0.9

print("*" * 100)
print("Evaluating TIES-Merging.")
ft_accuracies_path = os.path.join(args.save, f"ties_ft_accuracies.json")
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
sign_vectors = []

for dataset in eval_datasets:
    pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

    task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    # TIES-Merging
    if sparsity > 0.0: # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
        with torch.no_grad():
            # Trim redundant params (according to global magnitude)
            global_scores = torch.cat([torch.flatten(v).abs() for v in task_vector.vector.values()])
            threshold, _ = torch.kthvalue(global_scores, int(sparsity * global_scores.numel()))
            sgn = {}
            for key in task_vector.vector:
                score = task_vector.vector[key].abs()
                task_vector.vector[key].mul_(torch.where(score <= threshold, 0.0, 1.0))
                # Store signs
                sgn[key] = torch.sign(task_vector.vector[key])

    task_vectors.append(task_vector)
    sign_vectors.append(sgn)

with torch.no_grad():
    # Elect final sign
    agg_task_vector = {}
    for vect in task_vectors:
        for key in vect.vector:
            if key not in agg_task_vector:
                agg_task_vector[key] = vect.vector[key].clone()
            else:
                agg_task_vector[key] += vect.vector[key].clone()

    majority_sign = torch.sign(torch.cat([torch.flatten(v).abs() for v in agg_task_vector.values()]).sum())

    # Disjoint merge
    non_zero_counts = {}
    disjoint_agg = {}
    for vect in task_vectors:
        for key in vect.vector:
            sgn_m = torch.sign(agg_task_vector[key])
            sgn_m[sgn_m == 0] = majority_sign

            rows_to_keep = torch.where(sgn_m > 0, vect.vector[key] > 0, vect.vector[key] < 0)
            selected_entries = vect.vector[key] * rows_to_keep

            if key not in non_zero_counts:
                non_zero_counts[key] = (selected_entries != 0).float()
                disjoint_agg[key] = selected_entries
            else:
                non_zero_counts[key] += (selected_entries != 0).float()
                disjoint_agg[key] += selected_entries

    for key in non_zero_counts:
        disjoint_agg[key] /= torch.clamp(non_zero_counts[key], min=1)
    
    task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    for key in task_vector.vector:
        task_vector.vector[key].copy_(disjoint_agg[key])

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

save_file = f"{args.save}/ties_{sparsity}_additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)