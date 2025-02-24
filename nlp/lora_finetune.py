import os
import time
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from T5Wrapper import T5Wrapper
from linearize import LinearizedT5Wrapper
#from distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from data.PytorchDataset import PytorchDataset
from data.Batcher import Batcher
from data.dataset_readers import get_datasetReader
from data.dataset_mixture import get_datasetMixtureReader
from eval.eval import eval_single_dataset
from args import parse_arguments

import re
import peft
from peft import PeftModel, prepare_model_for_kbit_training, PeftConfig, get_peft_model, LoraConfig, TaskType
from linearize import LinearizedModule

def finetune(rank, args):
    train_dataset = args.train_dataset
    ckpdir = os.path.join(args.save, train_dataset)

    assert args.finetuning_mode in [
        "lora",
    ], "Only lora fine-tuning is supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    # Check if checkpoints already exist
    ft_path = (
        os.path.join(args.save, train_dataset, "linear_finetuned.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "lora_finetuned.pt")
    )
    zs_path = (
        os.path.join(args.save, train_dataset, "linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(args.save, train_dataset, "lora_zeroshot.pt")
    )
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    assert train_dataset is not None, "Please provide a training dataset."

    if args.load is not None and args.load.endswith("pt"):
        raise ValueError('args.load not supported yet!')
    else:
        print("Building model and tokenizer.")
        if linearized_finetuning:
            model = LinearizedT5Wrapper(args, keep_lang=False)
            tokenizer = model.model.tokenizer
        else:
            pretrainedModel_name = f'google-t5/{args.model}'
            transformer = AutoModelForSeq2SeqLM.from_pretrained(pretrainedModel_name)
            tokenizer = AutoTokenizer.from_pretrained(pretrainedModel_name, model_max_length=args.max_seq_len)
            init_transformer = transformer
        
        # Freeze the original parameters
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

        peft_config = LoraConfig(
            # the task to train for (sequence-to-sequence language modeling in this case)
            task_type=TaskType.SEQ_2_SEQ_LM,
            # the dimension of the low-rank matrices
            r=16,
            # the scaling factor for the low-rank matrices
            lora_alpha=32,
            # the dropout probability of the LoRA layers
            lora_dropout=0.0,
            target_modules=["q","v"],
        )
        transformer = get_peft_model(transformer, peft_config)
        model = T5Wrapper(transformer, tokenizer)

        for name, layer in model.transformer.base_model.model.named_modules():
            if isinstance(layer, peft.tuners.lora.layer.Linear):
                name = re.sub(r'(\.)(\d+)(\.)', r'[\2].', name)
                exec(f'transformer.{name} = LinearizedModule(layer, init_transformer.{name})')

        if not os.path.exists(os.path.join(args.save, train_dataset)):
            os.makedirs(os.path.join(args.save, train_dataset))
        zs_name = 'linear_zeroshot.pt' if linearized_finetuning else 'lora_zeroshot.pt'
        model.save(os.path.join(args.save, train_dataset, zs_name))
    
    model = model.cuda()

    print_every = 100

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
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

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
    ft_name = 'linear_finetuned.pt' if linearized_finetuning else 'lora_finetuned.pt'
    model.save(os.path.join(args.save, train_dataset, ft_name))
        

if __name__ == '__main__':
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