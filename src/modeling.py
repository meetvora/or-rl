from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def validate_quantization_args(load_in_4bit: bool = False, load_in_8bit: bool = False) -> None:
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one quantization mode: --load_in_4bit or --load_in_8bit.")


def quantization_config(load_in_4bit: bool = False, load_in_8bit: bool = False):
    validate_quantization_args(load_in_4bit, load_in_8bit)
    if not load_in_4bit and not load_in_8bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError("4-bit/8-bit loading requires transformers with bitsandbytes support.") from exc
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_compute_dtype(),
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def _compute_dtype():
    import torch

    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def text_causal_lm_kwargs(
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch

    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION", "sdpa").strip()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "local_files_only": local_files_only,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    qconfig = quantization_config(load_in_4bit, load_in_8bit)
    if qconfig is not None:
        kwargs["quantization_config"] = qconfig
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return kwargs


def load_text_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

    local_files_only = str2bool(os.environ.get("HF_LOCAL_FILES_ONLY", "false"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_text_causal_lm(
    model_name_or_path: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    adapter: bool | None = None,
    local_files_only: bool | None = None,
    trainable_adapter: bool = False,
):
    from transformers import AutoModelForCausalLM

    if local_files_only is None:
        local_files_only = str2bool(os.environ.get("HF_LOCAL_FILES_ONLY", "false"))
    kwargs = text_causal_lm_kwargs(load_in_4bit, load_in_8bit, local_files_only)
    model_path = Path(model_name_or_path)
    is_adapter = (model_path / "adapter_config.json").exists() if adapter is None else adapter
    if is_adapter:
        try:
            from peft import AutoPeftModelForCausalLM
        except Exception as exc:
            raise RuntimeError("Loading adapter checkpoints requires peft.") from exc
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            is_trainable=trainable_adapter,
            **kwargs,
        )
        if trainable_adapter:
            enable_trainable_adapter_parameters(model)
        return model
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def enable_trainable_adapter_parameters(model) -> int:
    trainable = 0
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(token in lowered for token in ("lora_", "adapter", "modules_to_save")):
            if not parameter.requires_grad:
                parameter.requires_grad_(True)
            trainable += parameter.numel()
    return trainable


def prepare_lora_model_if_needed(model, enabled: bool, load_in_4bit: bool, load_in_8bit: bool):
    if not enabled:
        return model
    if load_in_4bit or load_in_8bit:
        try:
            from peft import prepare_model_for_kbit_training
        except Exception as exc:
            raise RuntimeError("k-bit LoRA training requires peft.") from exc
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )
    return model
