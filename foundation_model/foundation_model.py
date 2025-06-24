#!/usr/bin/env python3
"""
Medical Foundation Model using Kolmogorov-Arnold Network (KAN)

This script implements a foundational base model capable of both segmentation
and classification tasks using the Kolmogorov-Arnold Network architecture.
"""

import os

# Set tokenizers parallelism to false before other imports to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import torch
from trainers.kan_trainer import KANFoundationTrainer


def main():
    parser = argparse.ArgumentParser(description="Medical Foundation Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to resume checkpoint, if any")
    parser.add_argument("--mode", type=str, choices=["train","finetune","test"], default="train", help="Operation mode: train, finetune, or test")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Use CPU if specified or if CUDA is not available
    if args.device == "cpu" or not torch.cuda.is_available():
        device = "cpu"
        print("Using CPU for training")
    else:
        device = "cuda"
        print(f"Using GPU for training: {torch.cuda.get_device_name(0)}")

    # Create trainer
    trainer = KANFoundationTrainer(config, device)

    # Resume or preload checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Execute mode
    if args.mode == "train":
        trainer.train_all()
    elif args.mode == "finetune":
        # Preload checkpoint for finetuning if not in config
        if args.resume:
            pass  # resume already loaded
        trainer.finetune_all()
    elif args.mode == "test":
        # Load checkpoint and run test
        if args.resume:
            pass
        # Implement test logic or reuse finetune_all for evaluation
        print("Test mode not implemented yet.")

if __name__ == "__main__":
    main()
