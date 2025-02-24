import json

import torch.backends.cuda
from args import parse_arguments
from eval import eval_single_dataset
from linearize import LinearizedImageEncoder
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

accuracies = {}

print("*" * 100)
print("Evaluating Breadcrumbs models.")

for sparsity in [0.8]:
    # sparsity := percentage of weights removed
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
        print(f"Evaluating Breadcrumbs on {dataset} - {sparsity} sparsity")

        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

        try:
            task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue

        # Breadcrumbs
        if sparsity > 0.0: # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
            with torch.no_grad():
                top_k_keep = 0.2
                top_k_remove = 0.1
                for key in task_vector.vector:
                    if any(x in key for x in ['attn', 'mlp', 'conv']):
                        tensor = task_vector.vector[key]
                        # Remove top
                        top_k_int = int(tensor.shape[-1] * top_k_remove)
                        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
                        mask = torch.ones_like(tensor)
                        mask.scatter_(len(tensor.shape) - 1, masked_indices, 0.0)
                        tensor.mul_(mask)

                        # Keep top
                        top_k_int = int(tensor.shape[-1] * top_k_keep)
                        _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
                        mask = torch.zeros_like(tensor)
                        mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)
                        tensor.mul_(mask)

        # Apply sparsified task vector
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

        for split in ["test", "val"]:
            # Evaluate
            print("=" * 100)
            print(f"Evaluating on {split} split.")
            eval_dataset = dataset if split == "test" else f"{dataset}Val"

            accuracies[eval_dataset] = eval_single_dataset(
                image_encoder, eval_dataset, args
            )["top1"]

# Save results
save_path = f"{args.save}/breadcrumbs_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)