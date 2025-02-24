import json

import torch.backends.cuda
from args import parse_arguments
from eval import eval_single_dataset
from linearize import LinearizedImageEncoder
from task_vectors import LinearizedTaskVector, NonLinearTaskVector, LinLoRATaskVector

args = parse_arguments()

accuracies = {}

print("*" * 100)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
print("Evaluating linear LoRA models.")

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
    
    pretrained_checkpoint = f"{args.save}/{dataset}Val/linlora_zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/linlora_finetuned.pt"

    task_vector = LinLoRATaskVector(pretrained_checkpoint, finetuned_checkpoint)

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
save_path = f"{args.save}/linlora_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
