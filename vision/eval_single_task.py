import json

import os
import torch.backends.cuda
from args import parse_arguments
from eval import eval_single_dataset
from linearize import LinearizedImageEncoder
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

accuracies = {}
POSTHOC = False

print("*" * 100)
if POSTHOC: #args.finetuning_mode == "posthoc":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("Evaluating post-hoc linearized models.")
elif args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode in ['linear', 'linear_cossim']:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("Evaluating linear FT models.")
else:
    print(f"Evaluating {args.finetuning_mode} models.")

for dataset in [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]:
    print("*" * 100)
    print(f"Evaluating on {dataset}")
    
    pretrained_checkpoint = (
        f"{args.save}/{dataset}Val/zeroshot.pt"
        if args.finetuning_mode == "standard"
        else f"{args.save}/{dataset}Val/{args.finetuning_mode}_zeroshot.pt"
        #else f"{args.save}/{dataset}Val/zeroshot.pt"
    )

    finetuned_checkpoint = (
        f"{args.save}/{dataset}Val/finetuned.pt"
        if args.finetuning_mode == "standard"
        else f"{args.save}/{dataset}Val/{args.finetuning_mode}_finetuned.pt"
    )

    if args.finetuning_mode == "none":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        try:
            task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue
    else:
        try:
            task_vector = (
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
                if args.finetuning_mode in ['linear', 'linear_cossim']
                else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue

    if args.finetuning_mode == "none":
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    elif POSTHOC:
        zs_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        ft_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        image_encoder = LinearizedImageEncoder(
            init_encoder=zs_encoder, image_encoder=ft_encoder, args=args
        )
    else:
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

    for split in ["test", "val"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            image_encoder, eval_dataset, args
        )["top1"]


if args.finetuning_mode == "none":
    # Evaluate zero-shot accuracy on ImageNet
    for split in ["ImageNetVal", "ImageNet"]:
        accuracies[split] = eval_single_dataset(image_encoder, split, args)["top1"]

# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_accuracies.json"
elif POSTHOC: #args.finetuning_mode == "posthoc":
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
