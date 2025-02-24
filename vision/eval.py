import numpy as np
import torch
import tqdm

import torch.nn.functional as F
import utils
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from heads import get_classification_head
from linearize import LinearizedImageEncoder
from modeling import ImageClassifier
from sklearn.linear_model import RidgeClassifier

cached_loaders = {}

def eval_single_dataset(image_encoder, dataset_name, args):

    if dataset_name not in cached_loaders:
        dataset = get_dataset(
            dataset_name,
            image_encoder.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=4
        )
        dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
        cached_loaders[dataset_name] = dataloader
    else:
        dataloader = cached_loaders[dataset_name]

    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)
    model.cuda()
    model.eval()

    #model = model.to(torch.bfloat16)

    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)#.to(torch.bfloat16)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%")

    return metrics


def eval_single_dataset_train_split(image_encoder, dataset_name, args):

    dataset = get_dataset(
        dataset_name if 'Val' in dataset_name else f'{dataset_name}Val',
        image_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=4
    )
    dataloader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)

    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)
    model.cuda()
    model.eval()

    #model = model.to(torch.bfloat16)

    device = args.device

    with torch.no_grad():
        loss, top1, correct, n = 0.0, 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)#.to(torch.bfloat16)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)
            loss += F.cross_entropy(logits, y).item()
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n
        loss = loss / n

    metrics = {"top1": top1, "avg_loss": loss}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}% | Train Loss: {loss}")

    return metrics


def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )
    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)
        #results = eval_single_dataset_train_split(image_encoder, dataset_name, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    task_vector, pretrained_checkpoint, args, scaling_coef, posthoc_linearization=False
):
    image_encoder = task_vector.apply_to(
        pretrained_checkpoint, scaling_coef=scaling_coef
    )
    if posthoc_linearization:
        pretrained_encoder = task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=0.0
        )
        image_encoder = LinearizedImageEncoder(
            init_encoder=pretrained_encoder, image_encoder=image_encoder, args=args
        )
    coef_info = evaluate(image_encoder, args)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info


def evaluate_task_vector(
    task_vector, pretrained_checkpoint, args, posthoc_linearization=False
):
    info = {}

    for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_checkpoint,
            args,
            scaling_coef,
            posthoc_linearization,
        )

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results


def nonlinear_advantage(acc_linear, acc_nonlinear, num_classes):
    err_linear = 1 - acc_linear
    err_nonlinear = 1 - acc_nonlinear
    return (err_linear - err_nonlinear) * num_classes / (num_classes - 1)