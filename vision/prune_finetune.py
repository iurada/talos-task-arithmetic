import os
import time

import torch

from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from eval import eval_single_dataset
from heads import get_classification_head
from linearize import LinearizedImageEncoder
from modeling import ImageClassifier, ImageEncoder
from utils import LabelSmoothing, cosine_lr
import layers
import pruners
import optimizers
import tqdm
import torch.nn.functional as F
from task_vectors import NonLinearTaskVector
import copy

def finetune(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    # Check if checkpoints already exist
    ft_path = (
        os.path.join(args.save, train_dataset, "finetuned.pt")
        if args.finetuning_mode == "standard"
        else os.path.join(args.save, train_dataset, f"{args.finetuning_mode}_finetuned.pt")
    )
    zs_path = (
        os.path.join(args.save, train_dataset, "zeroshot.pt")
        if args.finetuning_mode == "standard"
        else os.path.join(args.save, train_dataset, f"{args.finetuning_mode}_zeroshot.pt")
    )
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None:
        print(f'[LOADING] {args.load}')
        pt_ckpt = f'{args.save}/{train_dataset}/{args.load}zeroshot.pt'
        ft_ckpt = f'{args.save}/{train_dataset}/{args.load}finetuned.pt'
        image_encoder = NonLinearTaskVector(pt_ckpt, ft_ckpt).apply_to(pt_ckpt, 1.0)
    else:
        print("Building image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    classification_head = get_classification_head(args, train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    #! Prune ==============
    preprocess_fn = model.train_preprocess
    print_every = 100
    prune_bs = 64
    if args.model == 'ViT-B-16':
        prune_bs = 16
    if args.model == 'ViT-L-14':
        prune_bs = 2

    # Using Val split as pruning set
    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=prune_bs,
        num_workers=4,
    )
    data_loader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    ROUNDS = 4
    N_PRUNING_BATCHES = -1

    layers.mask_pretrained_vit(model, args.device, torch.float32, skip_ln=False)

    model = model.to(args.device)
    
    pruner = pruners.TaLoS(pruners.masked_parameters(model))

    sparsity = 1.0 - args.sparsity
    zeros_thresh = 1.0
    pruner.R = args.R

    for layer in model.modules():
        if hasattr(layer, 'masking'):
            layer.masking = True
    
    if sparsity < 1.0:
        for round in range(ROUNDS):
            sparse = sparsity**((round + 1) / ROUNDS)
            print('[+] Target sparsity:', sparse)
            pruner.score(model, None, data_loader, args.device, N_PRUNING_BATCHES)
            mode = 'global_copy'
            pruner.mask(sparse, mode)

    remaining_params, total_params = pruner.stats()
    print(f'{int(remaining_params)} / {int(total_params)} | {remaining_params / total_params}')
    
    with torch.no_grad():
        tot, cnt = 0, 0
        for layer in model.modules():
        
            if hasattr(layer, 'masking'):
                layer.masking = False
    
                for k in layer._buffers:
                    if 'mask' in k and layer._buffers[k] is not None:
                        layer._buffers[k] = layer._buffers[k].cpu()
                        layer._buffers[k] = None
                        
                for name, param in layer.named_parameters():
                    if hasattr(param, 'score'):
                        zeros_pctg = param.score[param.score == 0.0].numel() / param.score.numel()
                        if zeros_pctg >= zeros_thresh:
                            param.requires_grad_(False)
                            param.score = param.score.to('cpu')
                            delattr(param, 'score')
                            cnt += 1
                        else:
                            param.score = param.score.to(args.device)
                        tot += 1
                        
            elif len(list(layer.children())) == 0:
                for name, param in layer.named_parameters():
                    param.requires_grad_(False)
    
        print(f'Frozen {cnt} / {tot} params. ({100 * cnt / tot:.2f}%)')

    torch.cuda.empty_cache()

    #! Train ==============
    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=5,
    )
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    num_batches = len(dataset.train_loader)

    ddp_loader = distribute_loader(data_loader)
    ddp_model = model

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]

    print("=" * 100)
    print("Using [AdaptW] Optimizer")
    print("=" * 100)
    optimizer = optimizers.AdaptW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Saving zero-shot model
    if args.save is not None and is_main_process():
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, "zeroshot.pt")
            if args.finetuning_mode == "standard"
            else os.path.join(ckpdir, f"{args.finetuning_mode}_zeroshot.pt")
        )
        ddp_model.image_encoder.save(model_path)

    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()
            data_time = time.time() - start_time

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                logits = ddp_model(inputs)
                loss = loss_fn(logits, labels) / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)

                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = (
                    os.path.join(ckpdir, f"checkpoint_{step}.pt")
                    if args.finetuning_mode == "standard"
                    else os.path.join(ckpdir, f"{args.finetuning_mode}_checkpoint_{step}.pt")
                )
                ddp_model.module.image_encoder.save(model_path)

            if (
                i % print_every == 0
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )

    if is_main_process():
        image_encoder = ddp_model.image_encoder
        eval_single_dataset(image_encoder, train_dataset, args)

    if args.save is not None and is_main_process():
        zs_path = (
            os.path.join(ckpdir, f"zeroshot.pt")
            if args.finetuning_mode == "standard"
            else os.path.join(ckpdir, f"{args.finetuning_mode}_zeroshot.pt")
        )
        ft_path = (
            os.path.join(ckpdir, f"finetuned.pt")
            if args.finetuning_mode == "standard"
            else os.path.join(ckpdir, f"{args.finetuning_mode}_finetuned.pt")
        )
        image_encoder.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == "__main__":
    train_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN"
    ]
    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
    }

    for dataset in train_datasets:
        args = parse_arguments()

        # HACK: Some command line arguments are overwritten by defaults here.
        args.lr = 1e-5
        args.epochs = epochs[dataset]
        args.train_dataset = dataset + "Val"
        args.batch_size = 128
        args.num_grad_accumulation = 1

        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
