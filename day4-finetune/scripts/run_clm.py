"""
run_clm.py — QLoRA fine-tuning of a causal LM on SageMaker.

Runs inside the HuggingFace training container on an ml.g5.12xlarge
(4× A10G, 96 GB total VRAM).

Pipeline:
  1. Load base model in 4-bit NF4 (frozen)
  2. Find all Linear4bit layers → LoRA injection points
  3. Attach LoRA adapters (r=16, alpha=32) in BF16
  4. Apply Day 3's mixed-precision policy (norms in FP32, rest in BF16)
  5. Train with HuggingFace Trainer
  6. Merge LoRA adapters back into base weights
  7. Save merged model to /opt/ml/model (SageMaker auto-uploads to S3)
"""

import os
import argparse

import torch
import bitsandbytes as bnb
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
from peft.tuners.lora import LoraLayer
from huggingface_hub import login


# ---------------------------------------------------------------------
# 1. Argument parsing — SageMaker passes hyperparameters as CLI args
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--merge_weights", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------
# 2. Find all LoRA injection points at runtime
# ---------------------------------------------------------------------
def find_all_linear_names(model):
    """
    Every linear layer loaded in 4-bit becomes a bnb.nn.Linear4bit.
    Those are exactly the layers where LoRA adapters can attach.
    """
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            parts = name.split(".")
            lora_module_names.add(parts[-1] if len(parts) > 1 else parts[0])
    # LM head is excluded from LoRA by convention — it's not a transformer block
    lora_module_names.discard("lm_head")
    return list(lora_module_names)


# ---------------------------------------------------------------------
# 3. Build the PEFT model with LoRA + Day 3's mixed-precision policy
# ---------------------------------------------------------------------
def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = find_all_linear_names(model)
    print(f"Found {len(target_modules)} modules to quantize: {target_modules}")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Day 3's mixed-precision policy applied layer by layer
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)       # LoRA adapters in BF16
        if "norm" in name:
            module = module.to(torch.float32)            # Norms in FP32 (stability)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)   # LM head + embeddings in BF16

    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------
# 4. Main training function
# ---------------------------------------------------------------------
def train(args):
    set_seed(args.seed)

    # Log in to Hugging Face (required for gated models like Mistral)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load the pre-tokenized, pre-chunked dataset from Day 2
    dataset = load_from_disk(args.dataset_path)
    print(f"Loaded dataset with {len(dataset)} chunks")

    # QLoRA quantization config (Day 3 math → code)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load the base model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        use_cache=False if args.gradient_checkpointing else True,
        device_map="auto",
        force_download=True,        # workaround for flaky downloads on first run
    )

    # Attach LoRA + apply mixed-precision policy
    model = create_peft_model(
        model,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
    )

    # Tokenizer — we need it for the final save
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # ---- Training ----
    output_dir = "/tmp/mistral"     # checkpoint scratch space

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",         # cheap but no crash recovery (production: "steps")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    # ---- Save ----
    sagemaker_save_dir = "/opt/ml/model/"   # SageMaker auto-uploads this to S3

    if args.merge_weights:
        # Step 1: save LoRA adapters to a temp location
        trainer.model.save_pretrained(output_dir, safe_serialization=False)

        # Step 2: free GPU memory before re-loading for merge
        del model, trainer
        torch.cuda.empty_cache()

        # Step 3: reload base + adapter, merge into one set of weights
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        model = model.merge_and_unload()

        # Step 4: save merged model (this is what Day 5 will deploy)
        model.save_pretrained(
            sagemaker_save_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
    else:
        # If not merging, just save the adapter
        trainer.model.save_pretrained(sagemaker_save_dir, safe_serialization=True)

    # Save tokenizer alongside the model for inference
    tokenizer.save_pretrained(sagemaker_save_dir)
    print(f"Model saved to {sagemaker_save_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
