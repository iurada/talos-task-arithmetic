import json

import os
import torch.backends.cuda
from args import parse_arguments
from eval.eval import eval_single_dataset
from linearize import LinearizedT5Wrapper
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

accuracies = {}
POSTHOC = False

print("*" * 100)
if POSTHOC:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("Evaluating post-hoc linearized models.")
    args.max_seq_len = 512
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
    print(f"Evaluating on {dataset}")
    
    pretrained_checkpoint = (
        f"{args.save}/{dataset}/zeroshot.pt"
        if args.finetuning_mode == "standard"
        else f"{args.save}/{dataset}/{args.finetuning_mode}_zeroshot.pt"
    )

    finetuned_checkpoint = (
        f"{args.save}/{dataset}/finetuned.pt"
        if args.finetuning_mode == "standard"
        else f"{args.save}/{dataset}/{args.finetuning_mode}_finetuned.pt"
    )

    if args.finetuning_mode == "none":
        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        try:
            task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue
    else:
        try:
            task_vector = (
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
                if "linear" in args.finetuning_mode
                else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue

    if args.finetuning_mode == "none":
        model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    elif POSTHOC:
        zs_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        ft_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        model = LinearizedT5Wrapper(
            init_transformer=zs_encoder.transformer, transformer=ft_encoder.transformer, args=args
        )
    else:
        model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

    for split in ["test", "validation"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            split, model, model.tokenizer, dataset, args
        )["top1"]


if args.finetuning_mode == "none":
    # Evaluate zero-shot accuracy on COPA (task: causal reasoning)
    control_dataset = 'rte'
    print("*" * 100)
    print(f"Evaluating on {control_dataset}")
    for split in ["test", "validation"]:
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = control_dataset if split == "test" else f"{control_dataset}Val"
        accuracies[eval_dataset] = eval_single_dataset(
            split, model, model.tokenizer, control_dataset, args
        )["top1"]

# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_accuracies.json"
elif POSTHOC:
    save_path = f"{args.save}/posthoc_{args.finetuning_mode}_ft_accuracies.json"
elif args.finetuning_mode == "standard":
    save_path = f"{args.save}/ft_accuracies.json"
elif args.finetuning_mode == "linear":
    save_path = f"{args.save}/linear_ft_accuracies.json"
else:
    save_path = f"{args.save}/{args.finetuning_mode}_ft_accuracies.json"

if os.path.exists(save_path):
    with open(save_path) as f:
        old_accuracies = json.load(f)
    old_accuracies.update(accuracies)
    accuracies = old_accuracies
    
with open(save_path, "w") as f:
    json.dump(accuracies, f)
