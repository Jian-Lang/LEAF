from argparse import ArgumentParser
from pathlib import Path

import torch
import unsloth
from datasets import Dataset
from icecream import ic
from loguru import logger
from PIL import Image
from transformers import Qwen2_5_VLModel, TextStreamer
from trl import SFTConfig, SFTTrainer
from unsloth import (
    FastVisionModel,  # FastLanguageModel for LLMs
    is_bf16_supported,
)
from unsloth.trainer import UnslothVisionDataCollator

from distill.generate_dataset.generate_dataset import (
    convert_to_conversation,
    generate_dataset_explain,
    generate_dataset_reason,
)


def finetune(
    dataset_name, stage, model_name, load_path, save_path, dataset_cfg, peft_cfg, trainer_cfg, **kargs
):
    if stage == "reason":
        train_dataset = Dataset.from_generator(lambda: generate_dataset_reason(dataset_name, **dataset_cfg))
    elif stage == "explain":
        train_dataset = Dataset.from_generator(lambda: generate_dataset_explain(dataset_name, **dataset_cfg))
    elif stage == "explain-label":
        train_dataset = Dataset.from_generator(
            lambda: generate_dataset_explain(dataset_name, use_label=True, **dataset_cfg)
        )
    converted_train_dataset = [convert_to_conversation(sample) for sample in train_dataset]
    # ic(converted_dataset[0])
    if load_path is None:
        load_path = model_name

    model, tokenizer = FastVisionModel.from_pretrained(
        load_path,
        max_seq_length=8192 * 2,
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )
    if load_path == model_name:
        model = FastVisionModel.get_peft_model(
            model,
            **peft_cfg,
        )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=converted_train_dataset,
        args=SFTConfig(
            **trainer_cfg,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            report_to="wandb",  # For Weights and Biases
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
        ),
    )
    trainer_stats = trainer.train(resume_from_checkpoint=None)
    save_path = save_path
    model.save_pretrained(save_path)  # Local saving
    tokenizer.save_pretrained(save_path)
    logger.info(f"Saved model to {save_path}")
