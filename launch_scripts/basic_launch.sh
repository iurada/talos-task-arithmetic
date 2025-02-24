# Fine-tuning
python vision/finetune.py \
--model=ViT-B-32 \
--finetuning-mode=standard \
--data-location=/path/to/datasets \
--save=/path/to/checkpoints

# Inference
python vision/eval_single_task.py \
--model=ViT-B-32 \
--finetuning-mode=standard \
--data-location=/path/to/datasets \
--save=/path/to/checkpoints

python vision/eval_task_addition.py \
--model=ViT-B-32 \
--finetuning-mode=standard \
--data-location=/path/to/datasets \
--save=/path/to/checkpoints

python vision/eval_task_negation.py \
--model=ViT-B-32 \
--finetuning-mode=standard \
--data-location=/path/to/datasets \
--save=/path/to/checkpoints