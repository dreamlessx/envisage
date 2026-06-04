"""WebDataset-based DataLoader for envisage training pairs.

Reads TAR shards produced by scripts/generate_pairs.py and yields
batches of (input, conditioning_depth, mask, target) tensors.

Each sample in the shard contains:
  - input.png:  degraded clinical input image
  - depth.png:  modified depth map (conditioning signal)
  - mask.png:   surgical mask (inpainting region)
  - target.png: TPS-warped target (surgical outcome)

All images are loaded as RGB tensors normalized to [0, 1].
Masks and depth maps are loaded as single-channel tensors.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """DataLoader configuration."""

    shard_dir: str = "data/shards"
    shard_pattern: str = "*.tar"
    batch_size: int = 4
    num_workers: int = 4
    image_size: int = 512
    shuffle_buffer: int = 1000
    seed: int = 42


def _decode_image(data: bytes, mode: str = "RGB") -> torch.Tensor:
    """Decode PNG bytes to a normalized float32 tensor."""
    img = Image.open(io.BytesIO(data)).convert(mode)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]  # (1, H, W)
    else:
        arr = arr.transpose(2, 0, 1)  # (C, H, W)
    return torch.from_numpy(arr)


def _resize_tensor(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a (C, H, W) tensor to (C, size, size)."""
    return torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _process_sample(sample: dict, image_size: int) -> dict:
    """Process a single WebDataset sample into tensors."""
    input_img = _decode_image(sample["input.png"], mode="RGB")
    depth = _decode_image(sample["depth.png"], mode="L")
    mask = _decode_image(sample["mask.png"], mode="L")
    target = _decode_image(sample["target.png"], mode="RGB")

    # Resize all to target size
    if input_img.shape[-1] != image_size or input_img.shape[-2] != image_size:
        input_img = _resize_tensor(input_img, image_size)
        depth = _resize_tensor(depth, image_size)
        mask = _resize_tensor(mask, image_size)
        target = _resize_tensor(target, image_size)

    return {
        "input": input_img,
        "depth": depth,
        "mask": mask,
        "target": target,
        "__key__": sample.get("__key__", "unknown"),
    }


def create_dataloader(
    config: DataConfig | None = None,
) -> DataLoader:
    """Create a WebDataset DataLoader for training.

    Args:
        config: Data loading configuration.

    Returns:
        PyTorch DataLoader yielding batches of dicts with keys:
        input (B,3,H,W), depth (B,1,H,W), mask (B,1,H,W), target (B,3,H,W)
    """
    import webdataset as wds

    if config is None:
        config = DataConfig()

    shard_dir = Path(config.shard_dir)
    shards = sorted(shard_dir.glob(config.shard_pattern))

    if not shards:
        raise FileNotFoundError(
            f"No shards found matching {shard_dir / config.shard_pattern}"
        )

    shard_urls = [str(s) for s in shards]
    log.info("Loading %d shards from %s", len(shard_urls), shard_dir)

    image_size = config.image_size

    dataset = (
        wds.WebDataset(shard_urls)
        .shuffle(config.shuffle_buffer)
        .map(lambda s: _process_sample(s, image_size))
    )

    def collate_fn(batch: list[dict]) -> dict:
        return {
            "input": torch.stack([s["input"] for s in batch]),
            "depth": torch.stack([s["depth"] for s in batch]),
            "mask": torch.stack([s["mask"] for s in batch]),
            "target": torch.stack([s["target"] for s in batch]),
        }

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def verify_dataloader(config: DataConfig | None = None) -> None:
    """Quick verification that the DataLoader produces correct shapes."""
    if config is None:
        config = DataConfig(batch_size=2, num_workers=0)

    loader = create_dataloader(config)
    batch = next(iter(loader))

    print(f"Batch keys: {list(batch.keys())}")
    print(f"  input:  {batch['input'].shape} {batch['input'].dtype} [{batch['input'].min():.2f}, {batch['input'].max():.2f}]")
    print(f"  depth:  {batch['depth'].shape} {batch['depth'].dtype} [{batch['depth'].min():.2f}, {batch['depth'].max():.2f}]")
    print(f"  mask:   {batch['mask'].shape} {batch['mask'].dtype} [{batch['mask'].min():.2f}, {batch['mask'].max():.2f}]")
    print(f"  target: {batch['target'].shape} {batch['target'].dtype} [{batch['target'].min():.2f}, {batch['target'].max():.2f}]")

    bs = config.batch_size
    sz = config.image_size
    assert batch["input"].shape == (bs, 3, sz, sz), f"input shape mismatch: {batch['input'].shape}"
    assert batch["depth"].shape == (bs, 1, sz, sz), f"depth shape mismatch: {batch['depth'].shape}"
    assert batch["mask"].shape == (bs, 1, sz, sz), f"mask shape mismatch: {batch['mask'].shape}"
    assert batch["target"].shape == (bs, 3, sz, sz), f"target shape mismatch: {batch['target'].shape}"

    assert batch["input"].dtype == torch.float32
    assert 0.0 <= batch["input"].min() and batch["input"].max() <= 1.0

    print("DataLoader verification passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify envisage DataLoader")
    parser.add_argument("--shard-dir", type=str, default="data/shards")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    cfg = DataConfig(
        shard_dir=args.shard_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=0,
    )
    verify_dataloader(cfg)
