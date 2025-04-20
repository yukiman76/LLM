#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlined LLM Fine-tuning Script using TRL and PEFT
"""

import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)
from trl import SFTTrainer


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading a model"}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `huggingface-cli login`"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Override the default `torch.dtype` for model loading"}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention 2"},
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration"""
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use"},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)"},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file"},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization"},
    )
    prompt_column_name: Optional[str] = field(
        default="prompt",
        metadata={"help": "The name of the column in the datasets containing the prompts"},
    )
    response_column_name: Optional[str] = field(
        default="response",
        metadata={"help": "The name of the column in the datasets containing the responses"},
    )


@dataclass
class PeftArguments:
    """Arguments for Parameter-Efficient Fine-Tuning"""
    peft_mode: str = field(
        default="lora",
        metadata={"help": "The PEFT method to use. Options: 'lora', 'qlora'"},
    )
    lora_r: int = field(
        default=16, 
        metadata={"help": "Lora attention dimension"}
    )
    lora_alpha: int = field(
        default=32, 
        metadata={"help": "Lora alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05, 
        metadata={"help": "Lora dropout parameter"}
    )
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module names to target for PEFT"},
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    use_gradient_checkpointing: bool = field(
        default=True, 
        metadata={"help": "Whether to use gradient checkpointing"}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "Number of bits for quantization (4 or 8 for qlora)"},
    )
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether to use double quantization in QLoRA"},
    )


@dataclass
class CustomArguments:
    """Custom arguments for fine-tuning"""
    do_full_finetune: bool = field(
        default=False,
        metadata={"help": "Whether to do full fine-tuning with all parameters trainable"},
    )
    custom_logging_level: Optional[str] = field(
        default="INFO", 
        metadata={"help": "Logger level"}
    )
    nef_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "Noise alpha factor for NEFTune, if enabled"},
    )


def setup_logging(log_level: str = "INFO"):
    """Set up basic logging"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=getattr(logging, log_level),
    )
    return logging.getLogger(__name__)


def setup_model_and_tokenizer(model_args, peft_args, custom_args, logger):
    """Set up model and tokenizer with correct parameters for training"""
    # Convert dtype string to torch dtype
    torch_dtype = None
    if model_args.torch_dtype == "float32":
        torch_dtype = torch.float32
    elif model_args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif model_args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_args.torch_dtype == "auto":
        torch_dtype = "auto"

    # Load tokenizer
    tokenizer_name = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=model_args.use_auth_token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure model loading options
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    load_in_4bit = peft_args.peft_mode == "qlora" and peft_args.quantization_bit == 4
    load_in_8bit = peft_args.peft_mode == "qlora" and peft_args.quantization_bit == 8

    model_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_auth_token": model_args.use_auth_token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    # Configure quantization
    if load_in_4bit or load_in_8bit:
        model_kwargs["device_map"] = "auto"

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype or torch.float16,
                bnb_4bit_use_double_quant=peft_args.double_quantization,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif torch_dtype:
        model_kwargs["torch_dtype"] = torch_dtype

    # Use Flash Attention 2 if requested
    if model_args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    # Full fine-tuning or PEFT setup
    if custom_args.do_full_finetune:
        logger.info("Setting up for full fine-tuning (all parameters trainable)")
        # Explicitly mark all parameters as trainable
        for param in model.parameters():
            param.requires_grad = True
    else:
        # LoRA or QLoRA fine-tuning
        logger.info(f"Setting up PEFT with method: {peft_args.peft_mode}")
        
        # Prepare model for k-bit training if using quantization
        if load_in_4bit or load_in_8bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=peft_args.use_gradient_checkpointing
            )

        # Determine target modules based on model architecture if not specified
        if peft_args.target_modules is None:
            if "llama" in model_args.model_name_or_path.lower():
                peft_args.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "mistral" in model_args.model_name_or_path.lower():
                peft_args.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # Default for general transformer models
                peft_args.target_modules = ["query", "key", "value", "output"]
            logger.info(f"Automatically selected target modules: {peft_args.target_modules}")

        # Create and apply LoRA config
        peft_config = LoraConfig(
            r=peft_args.lora_r,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            bias=peft_args.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=peft_args.target_modules,
            # Make sure lm_head is trainable for standard LoRA
            modules_to_save=["lm_head"] if not load_in_4bit and not load_in_8bit else None,
        )
        
        # Apply PEFT to the model
        model = get_peft_model(model, peft_config)
        
        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {trainable_params}")
        
        # Critical fix: If no parameters are trainable, force LoRA parameters to be trainable
        if trainable_params == 0:
            logger.warning("No trainable parameters detected. Enabling gradients for LoRA layers...")
            # Make LoRA params trainable
            for name, param in model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
            # Recount trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"After fix: Number of trainable parameters: {trainable_params}")
            
            # If still no trainable parameters, abort
            if trainable_params == 0:
                raise ValueError("No trainable parameters found even after fixes. Check model configuration.")
        
        # Print trainable parameter info
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing if requested (saves memory)
    if peft_args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Required for gradient checkpointing

    return model, tokenizer


def load_and_prepare_datasets(data_args, tokenizer, logger):
    """Load and prepare datasets for training"""
    # Load dataset
    if data_args.dataset_name is not None:
        logger.info(f"Loading dataset: {data_args.dataset_name}")
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=None,
        )
    elif data_args.train_file is not None:
        logger.info(f"Loading dataset from files: {data_args.train_file}")
        data_files = {"train": data_args.train_file}
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file

        extension = data_args.train_file.split(".")[-1]
        if extension in ["json", "jsonl"]:
            dataset = load_dataset("json", data_files=data_files)
        elif extension == "csv":
            dataset = load_dataset("csv", data_files=data_files)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    else:
        raise ValueError("Dataset name or training file must be provided.")

    # Prepare the formatting function for dataset
    def format_dataset(examples):
        """Format the dataset examples for training"""
        prompt_column = data_args.prompt_column_name
        response_column = data_args.response_column_name
        
        formatted_examples = []
        for i in range(len(examples[prompt_column])):
            prompt = examples[prompt_column][i]
            response = examples[response_column][i]
            
            # Create formatted prompt-response pairs
            formatted_text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
            formatted_examples.append(formatted_text)
            
        return {"text": formatted_examples}
    
    # Apply formatting to datasets
    formatted_datasets = {}
    for split in dataset:
        formatted_datasets[split] = dataset[split].map(
            format_dataset, 
            batched=True,
            remove_columns=dataset[split].column_names
        )
    
    return formatted_datasets


def main():
    """Main training function"""
    # Parse arguments - avoid parameter conflicts with completely unique names
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, PeftArguments, TrainingArguments, CustomArguments)
    )
    model_args, data_args, peft_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logger = setup_logging(custom_args.custom_logging_level)
    logger.info(f"Starting fine-tuning process")

    # Set seed for reproducibility
    set_seed(training_args.seed)
    training_args.report_to = None
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_args, peft_args, custom_args, logger
    )

    # Verify trainable parameters - critical check
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        logger.error("No parameters require gradients! Training would fail.")
        raise ValueError("No trainable parameters found in the model.")
    else:
        logger.info(f"Total trainable parameters: {trainable_params}")

    # Prepare datasets
    formatted_datasets = load_and_prepare_datasets(data_args, tokenizer, logger)
    
    train_dataset = formatted_datasets["train"]
    eval_dataset = formatted_datasets.get("validation", None)

    # Setup trainer
    logger.info("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save model and tokenizer
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()