import json

import torch.backends.cuda
from args import parse_arguments
from eval import eval_single_dataset
from linearize import LinearizedImageEncoder
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

accuracies = {}

print("*" * 100)
print("Evaluating Non-linear FT models.")


for sparsity in [0.9, 0.95, 0.99]:
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
        print(f"Evaluating TIES-Merging on {dataset} - {sparsity} sparsity")

        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

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
                    if any(x in key for x in ['attn', 'mlp', 'conv']):
                        # Trim redundant params (according to global magnitude)
                        score = task_vector.vector[key].abs()
                        task_vector.vector[key].mul_(torch.where(score <= threshold, 0.0, 1.0))

        # Apply sparsified task vector
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

        for split in ["test", "val"]:
            # Evaluate
            print("=" * 100)
            print(f"Evaluating on {split} split.")
            eval_dataset = dataset if split == "test" else f"{dataset}Val"

            accuracies[eval_dataset + f'_{sparsity}'] = eval_single_dataset(
                image_encoder, eval_dataset, args
            )["top1"]

# Save results
save_path = f"{args.save}/ties_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)