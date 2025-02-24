import json

import torch.backends.cuda
from args import parse_arguments
from eval import eval_single_dataset
from linearize import LinearizedImageEncoder
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

accuracies = {}

print("*" * 100)

for lam in [0.2]:
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
        print(f"Evaluating TALL Mask / Consensus on {dataset} - {lam} lambda")

        # Construct TALL Mask
        multitask_vector = []
        for task_dataset in [
                "Cars",
                "DTD",
                "EuroSAT",
                "GTSRB",
                "MNIST",
                "RESISC45",
                "SUN397",
                "SVHN",
            ]:
            pretrained_checkpoint = f"{args.save}/{task_dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task_dataset}Val/finetuned.pt"
            multitask_vector.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))
        multitask_vector = sum(multitask_vector)

        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

        with torch.no_grad():
            tot, cnt = 0, 0
            for key in task_vector.vector:
                msk = torch.where(task_vector.vector[key].abs() > ((multitask_vector.vector[key] - task_vector.vector[key]) * lam), 1.0, 0.0)
                task_vector.vector[key].mul_(msk)
                tot += msk.numel()
                cnt += msk.sum().item()
        print(f'Sparsity: {100 * (1 - cnt / tot):.2f} %')

        # Apply sparsified task vector
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

        for split in ["test", "val"]:
            # Evaluate
            print("=" * 100)
            print(f"Evaluating on {split} split.")
            eval_dataset = dataset if split == "test" else f"{dataset}Val"

            accuracies[eval_dataset + f'_{lam}'] = eval_single_dataset(
                image_encoder, eval_dataset, args
            )["top1"]

# Save results
save_path = f"{args.save}/tallmask_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)