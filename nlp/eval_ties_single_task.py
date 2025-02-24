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
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("Evaluating post-hoc linearized models.")
elif args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "linear":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
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
    print(f"Evaluating TIES-Merging on {dataset} - {sparsity} sparsity")
    
    pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"

    try:
        task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue

    # TIES-Merging
    if sparsity > 0.0: # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
        with torch.no_grad():
            global_scores = torch.cat([torch.flatten(v).abs() for v in task_vector.vector.values()])
            threshold, _ = torch.kthvalue(global_scores, int(sparsity * global_scores.numel()))
            for key in task_vector.vector:
                # Trim redundant params (according to global magnitude)
                score = task_vector.vector[key].abs()
                task_vector.vector[key].mul_(torch.where(score <= threshold, 0.0, 1.0))

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
save_path = f"{args.save}/ties_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
