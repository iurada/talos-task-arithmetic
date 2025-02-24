import json

import torch.backends.cuda
from args import parse_arguments
from eval.eval import eval_single_dataset
#from linearize import LinearizedImageEncoder
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

accuracies = {}
POSTHOC = False

sparsity = 0.9

print("*" * 100)
if POSTHOC:
    print("Evaluating post-hoc linearized models.")
elif args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
else:
    print(f"Evaluating {args.finetuning_mode} models.")

for dataset in [
    'qasc',
    'wiki_qa',
    'quartz',
    'paws',
    'story_cloze',
    'winogrande',
    'wsc'
]:
    print("*" * 100)
    print(f"Evaluating DARE on {dataset} - {sparsity} sparsity")
    
    pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"

    try:
        task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue

    # Drop And REscale (DARE)
    if sparsity > 0.0: # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
        with torch.no_grad():
            for key in task_vector.vector:
                score = torch.randn_like(task_vector.vector[key])
                threshold, _ = torch.kthvalue(torch.flatten(score), k=int(sparsity * score.numel()))
                task_vector.vector[key].mul_(torch.where(score <= threshold, 0.0, 1.0 / (1.0 - sparsity)))

    model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    
    for split in ["test", "validation"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset + f'_{sparsity}'] = eval_single_dataset(
            split, model, model.tokenizer, dataset, args
        )["top1"]

# Save results
save_path = f"{args.save}/dare_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
