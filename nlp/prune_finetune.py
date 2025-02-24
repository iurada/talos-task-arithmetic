import os
import time

import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Model
from T5Wrapper import T5Wrapper
#from distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from data.PytorchDataset import PytorchDataset
from data.Batcher import Batcher
from data.dataset_readers import get_datasetReader
from eval.eval import eval_single_dataset
from args import parse_arguments
import layers
import pruners
import optimizers

def finetune(rank, args):
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

    if args.load is not None and args.load.endswith("pt"):
        raise ValueError('args.load not supported yet!')
    else:
        print("Building model and tokenizer.")
        pretrainedModel_name = f'google-t5/{args.model}'
        transformer = AutoModelForSeq2SeqLM.from_pretrained(pretrainedModel_name)
        tokenizer = AutoTokenizer.from_pretrained(pretrainedModel_name, model_max_length=args.max_seq_len)
        model = T5Wrapper(transformer, tokenizer)

    model = model.cuda()

    print_every = 100

    #! Prune ==============
    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": None,
    }

    dataset_reader = get_datasetReader(train_dataset, dataset_kwargs)
    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, 'cuda')
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=args.batch_size // 2,
        eval_batchSize=args.batch_size // 2,
        world_size=args.world_size,
        device=rank,
    )

    ROUNDS = 4
    N_PRUNING_BATCHES = -1
    SKIP_EMB = True
    SKIP_LN = False

    layers.mask_pretrained_t5(model, 'cuda', torch.float32, skip_ln=SKIP_LN, skip_emb=SKIP_EMB)
    model = model.to(args.device)
    
    pruner = pruners.TaLoS(pruners.masked_parameters(model))

    sparsity = 1.0 - args.sparsity #args.sparsity # weight remaining ratio i.e. how many weights to keep
    zeros_thresh = 1.0
    pruner.R = args.R

    for layer in model.modules():
        if hasattr(layer, 'masking'):
            layer.masking = True
    
    if sparsity < 1.0:
        for round in range(ROUNDS):
            sparse = sparsity**((round + 1) / ROUNDS)
            print('[+] Target sparsity:', sparse)
            pruner.score(model, None, batcher, args.device, N_PRUNING_BATCHES)
            pruner.mask(sparse, 'global_copy')
    
    remaining_params, total_params = pruner.stats()
    print(f'{int(remaining_params)} / {int(total_params)} | {remaining_params / total_params}')

    with torch.no_grad():
        tot, cnt = 0, 0
        for layer_name, layer in model.named_modules():
        
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
                            param.score.cpu()
                            delattr(param, 'score')
                            cnt += 1
                        tot += 1

            elif len(list(layer.children())) == 0:
                for name, param in layer.named_parameters():
                    param.requires_grad_(False)
    
        print(f'Frozen {cnt} / {tot} params. ({100 * cnt / tot:.2f}%)')

    #! Train ==============
    dataset_kwargs = {
        "few_shot_random_seed": None,
        "num_val_samples": 32,
        "max_datapoints_per_dataset_without_templates": None,
    }

    dataset_reader = get_datasetReader(train_dataset, dataset_kwargs)
    createPytorchDataset_fn = lambda dataset: PytorchDataset(dataset, tokenizer, 'cuda')
    batcher = Batcher(
        dataset_reader,
        createPytorchDataset_fn,
        train_batchSize=args.batch_size,
        eval_batchSize=args.batch_size * 2,
        world_size=args.world_size,
        device=rank,
    )
    train_iterator = batcher.get_trainBatches("train", template_idx=0)
    num_batches = args.num_batches

    params = [p for p in model.parameters() if p.requires_grad]
    print("=" * 100)
    print("Using [AdaptW] Optimizer")
    print("=" * 100)
    optimizer = optimizers.AdaptW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Saving zero-shot model
    if not os.path.exists(os.path.join(args.save, train_dataset)):
        os.makedirs(os.path.join(args.save, train_dataset))
    zs_name = 'linear_zeroshot.pt' if linearized_finetuning else f"{args.finetuning_mode}_zeroshot.pt"
    model.save(os.path.join(args.save, train_dataset, zs_name))

    model.train()
    for i in range(num_batches * args.num_grad_accumulation):
        start_time = time.time()

        train_batch = next(train_iterator)
        data_time = time.time() - start_time

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, current_metrics = model(train_batch)
            loss = loss / args.num_grad_accumulation
        scaler.scale(loss).backward()

        if (i + 1) % args.num_grad_accumulation == 0:
            # Take a gradient step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_time = time.time() - start_time

        if (
            i % print_every == 0
        ):
            percent_complete = 100 * i / num_batches
            print(
                f"Train Iteration: {i} [{percent_complete:.0f}% {i}/{num_batches}]\t"  # noqa: E501
                f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                flush=True,
            )

    # Eval
    eval_single_dataset('validation', model, tokenizer, train_dataset, args)
            
    # Save model
    if not os.path.exists(os.path.join(args.save, train_dataset)):
        os.makedirs(os.path.join(args.save, train_dataset))
    ft_name = 'linear_finetuned.pt' if linearized_finetuning else f"{args.finetuning_mode}_finetuned.pt"
    model.save(os.path.join(args.save, train_dataset, ft_name))


if __name__ == "__main__":
    train_datasets = [
        'qasc',       
        'wiki_qa',    
        'quartz',     
        'paws',       
        'story_cloze',
        'winogrande', 
        'wsc'         
    ]
    
    for dataset in train_datasets:
        args = parse_arguments()

        # HACK: Some command line arguments are overwritten by defaults here.
        args.lr = 1e-4
        args.wd = 0.0
        args.train_dataset = dataset

        args.batch_size = 64
        args.num_grad_accumulation = 16
        args.max_seq_len = 128
        args.num_batches = 75000

        print("=" * 100)
        print(f"Finetuning {args.model} on {dataset}")
        print("=" * 100)
        torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
