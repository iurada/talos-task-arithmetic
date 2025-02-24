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

sparsity = 0.9

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
print("Evaluating TIES-Merging models.")
ft_accuracies_path = os.path.join(args.save, f"ties_ft_accuracies.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    accs = json.load(f)
    new_accs = {}
    for k in accs:
        if 'wiki_qa' in k or 'story_cloze' in k:
            new_accs[k.split('_')[0] + '_' + k.split('_')[1]] = accs[k]
        else:
            new_accs[k.split('_')[0]] = accs[k]
    args.finetuning_accuracies = new_accs
    print(args.finetuning_accuracies)

control_dataset = "rte"
negation_accuracies = {}

for dataset in eval_datasets:
    pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
    task_vector = -NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    # TIES-Merging
    if sparsity > 0.0: # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
        with torch.no_grad():
            global_scores = torch.cat([torch.flatten(v).abs() for v in task_vector.vector.values()])
            threshold, _ = torch.kthvalue(global_scores, int(sparsity * global_scores.numel()))
            for key in task_vector.vector:
                # Trim redundant params (according to global magnitude)
                score = task_vector.vector[key].abs()
                task_vector.vector[key].mul_(torch.where(score <= threshold, 0.0, 1.0))

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

save_file = f"{args.save}/ties_{sparsity}_negations.json"
with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)