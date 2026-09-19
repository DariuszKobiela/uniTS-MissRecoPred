#!/usr/bin/env python3
# ruff: noqa: E402
"""Fine-tune the SD2 inpainting UNet on explicit windowed triplets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer

from reconstruction_models.sd2_pipeline import MODEL_ID_BASE


class WindowedInpaintingDataset(Dataset):
    """Load target, conditioning image and explicit inpainting mask."""

    def __init__(
        self,
        data_dir: Path,
        records: list[dict],
        prompt_dropout: float = 0.0,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.records = records
        self.prompt_dropout = prompt_dropout
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        clean = np.asarray(
            Image.open(self.data_dir / record["clean"]).convert("RGB"),
            dtype=np.float32,
        )
        condition = np.asarray(
            Image.open(self.data_dir / record["conditioning"]).convert("RGB"),
            dtype=np.float32,
        )
        mask = np.asarray(
            Image.open(self.data_dir / record["mask"]).convert("L"),
            dtype=np.float32,
        )

        clean_tensor = torch.from_numpy(clean).permute(2, 0, 1) / 127.5 - 1.0
        condition_tensor = torch.from_numpy(condition).permute(2, 0, 1) / 127.5 - 1.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor >= 0.5).float()
        masked_tensor = condition_tensor * (mask_tensor < 0.5)

        prompt = record["prompt"]
        if self.prompt_dropout > 0:
            rng = random.Random(self.seed + index)
            if rng.random() < self.prompt_dropout:
                prompt = ""

        return {
            "clean": clean_tensor,
            "masked": masked_tensor,
            "mask": mask_tensor,
            "prompt": prompt,
            "series_id": int(record["series_id"]),
            "encoding": record["encoding"],
        }


def read_manifest(path: Path, encodings: set[str] | None) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if encodings:
        records = [record for record in records if record["encoding"] in encodings]
    if not records:
        raise ValueError("No records selected from manifest")
    return records


def group_split(
    records: list[dict],
    validation_share: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split by base series so its four encodings never cross train/validation."""
    groups = sorted({int(record["series_id"]) for record in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    validation_count = max(1, int(round(len(groups) * validation_share)))
    validation_groups = set(groups[:validation_count])
    train = [record for record in records if int(record["series_id"]) not in validation_groups]
    validation = [record for record in records if int(record["series_id"]) in validation_groups]
    if not train or not validation:
        raise ValueError("Group split produced an empty train or validation set")
    return train, validation


def encode_prompts(
    prompts: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        return text_encoder(tokens.input_ids.to(device))[0]


def diffusion_loss(
    batch: dict,
    *,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    clean = batch["clean"].to(accelerator.device, dtype=weight_dtype)
    masked = batch["masked"].to(accelerator.device, dtype=weight_dtype)
    mask = batch["mask"].to(accelerator.device, dtype=weight_dtype)

    with torch.no_grad():
        clean_latents = vae.encode(clean).latent_dist.sample() * vae.config.scaling_factor
        masked_latents = vae.encode(masked).latent_dist.sample() * vae.config.scaling_factor
        text = encode_prompts(batch["prompt"], tokenizer, text_encoder, accelerator.device)

    latent_mask = F.interpolate(mask, size=clean_latents.shape[-2:], mode="nearest")
    noise = torch.randn_like(clean_latents)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (clean_latents.shape[0],),
        device=clean_latents.device,
        dtype=torch.long,
    )
    noisy = scheduler.add_noise(clean_latents, noise, timesteps)
    model_input = torch.cat([noisy, latent_mask, masked_latents], dim=1)
    prediction = unet(model_input, timesteps, text).sample

    if scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(clean_latents, noise, timesteps)
    else:
        target = noise
    return F.mse_loss(prediction.float(), target.float(), reduction="mean")


def validate(
    loader: DataLoader,
    **loss_kwargs,
) -> float:
    unet = loss_kwargs["unet"]
    accelerator = loss_kwargs["accelerator"]
    unet.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            with accelerator.autocast():
                loss = diffusion_loss(batch, **loss_kwargs)
            gathered = accelerator.gather_for_metrics(loss.detach().reshape(1))
            losses.extend(gathered.float().cpu().tolist())
    unet.train()
    return float(np.mean(losses))


def save_unet_checkpoint(
    output_dir: Path,
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    epoch: int,
    train_loss: float,
    validation_loss: float,
) -> Path:
    checkpoint = output_dir / "best_unet"
    checkpoint.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(checkpoint / "unet")
    (checkpoint / "training_state.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sd2_windowed_training")
    parser.add_argument("--output-dir", default="models/sd2_windowed")
    parser.add_argument("--model-id", default=MODEL_ID_BASE)
    parser.add_argument("--encodings", default="gaf,mtf,rp,spec")
    parser.add_argument("--validation-share", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prompt-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning SD2 requires a CUDA GPU")
    if not 0.0 < args.validation_share < 1.0:
        raise ValueError("--validation-share must be in (0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {item.strip() for item in args.encodings.split(",") if item.strip()}
    records = read_manifest(data_dir / "manifest.jsonl", selected)
    train_records, validation_records = group_split(records, args.validation_share, args.seed)

    train_dataset = WindowedInpaintingDataset(data_dir, train_records, args.prompt_dropout, args.seed)
    validation_dataset = WindowedInpaintingDataset(data_dir, validation_records, 0.0, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.train()
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=0.01)
    unet, optimizer, train_loader, validation_loader = accelerator.prepare(
        unet, optimizer, train_loader, validation_loader
    )

    loss_kwargs = {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "unet": unet,
        "scheduler": scheduler,
        "accelerator": accelerator,
        "weight_dtype": weight_dtype,
    }
    history = []
    best_validation = float("inf")
    patience = 0
    best_checkpoint = None

    for epoch in range(1, args.epochs + 1):
        unet.train()
        epoch_losses = []
        for batch in train_loader:
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    loss = diffusion_loss(batch, **loss_kwargs)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            epoch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(epoch_losses))
        validation_loss = validate(validation_loader, **loss_kwargs)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        history.append(record)
        accelerator.print(f"epoch={epoch} train={train_loss:.6f} validation={validation_loss:.6f}")

        if validation_loss < best_validation:
            best_validation = validation_loss
            patience = 0
            if accelerator.is_main_process:
                best_checkpoint = save_unet_checkpoint(
                    output_dir,
                    accelerator,
                    unet,
                    epoch,
                    train_loss,
                    validation_loss,
                )
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                accelerator.print("Early stopping")
                break
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        best_unet = UNet2DConditionModel.from_pretrained(best_checkpoint / "unet")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            args.model_id,
            unet=best_unet,
            safety_checker=None,
        )
        final_dir = output_dir / "best_model"
        pipeline.save_pretrained(final_dir)
        report = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_model": args.model_id,
            "data_dir": str(data_dir),
            "train_triplets": len(train_records),
            "validation_triplets": len(validation_records),
            "encodings": sorted(selected),
            "best_validation_loss": best_validation,
            "history": history,
            "best_model": str(final_dir),
            "split": "grouped by series_id",
        }
        (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Best model saved in {final_dir}")


if __name__ == "__main__":
    main()
