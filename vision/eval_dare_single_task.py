import json

import torch.backends.cuda
from args import parse_arguments
from eval import eval_single_dataset
from linearize import LinearizedImageEncoder
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

accuracies = {}

print("*" * 100)
print("Evaluating DARE models.")

for sparsity in [0.9]: #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]: #[0.1 * n for n in range(0, 10)] + [0.99]:
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
        print(f"Evaluating DARE on {dataset} - {sparsity} sparsity")

        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

        try:
            task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue

        # Drop And REscale (DARE)
        if sparsity > 0.0: # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
            with torch.no_grad():
                for key in task_vector.vector:
                    if any(x in key for x in ['attn', 'mlp', 'conv']):
                        score = torch.randn_like(task_vector.vector[key])
                        threshold, _ = torch.kthvalue(torch.flatten(score), k=int(sparsity * score.numel()))
                        task_vector.vector[key].mul_(torch.where(score <= threshold, 0.0, 1.0 / (1.0 - sparsity)))

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
save_path = f"{args.save}/dare_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)