import json

import torch.backends.cuda
from args import parse_arguments
from eval.eval import eval_single_dataset
#from linearize import LinearizedImageEncoder
from task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

accuracies = {}

print("*" * 100)

for lam in [0.6]:
    # sparsity := percentage of weights removed
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
        print(f"Evaluating TALL Mask / Consensus on {dataset} - {lam} lambda")

        # Construct TALL Mask
        multitask_vector = []
        for task_dataset in [
                'qasc',
                'wiki_qa',
                'quartz',
                'paws',
                'story_cloze',
                'winogrande',
                'wsc'
            ]:
            pretrained_checkpoint = f"{args.save}/{task_dataset}/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{task_dataset}/finetuned.pt"
            multitask_vector.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))
        multitask_vector = sum(multitask_vector)

        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
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
        model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

        for split in ["test", "validation"]:
            # Evaluate
            print("=" * 100)
            print(f"Evaluating on {split} split.")
            eval_dataset = dataset if split == "test" else f"{dataset}Val"

            accuracies[eval_dataset + f'_{lam}'] = eval_single_dataset(
                split, model, model.tokenizer, dataset, args
            )["top1"]

# Save results
save_path = f"{args.save}/tallmask_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)