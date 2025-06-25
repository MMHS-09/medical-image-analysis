import os
import csv

# Set tokenizers parallelism to false before other imports to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict
import logging
from pathlib import Path
import sys
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our custom modules
from models import create_medical_foundation_model
from datasets.loaders import MultiDatasetLoader
from core.losses import CombinedLoss
from utils.visualization import Visualizer
from utils.reporting import MetricsCollector, ReportGenerator
from utils.training import train_epoch, validate, compute_loss, compute_metrics


class KANFoundationTrainer:
    """
    Trainer for the KAN-based foundation model for medical imaging.

    This trainer implements sequential training on multiple datasets,
    optimizing a single foundation model that can perform both
    classification and segmentation tasks.
    """

    def __init__(self, config: Dict, device="cuda"):
        self.config = config
        self.device = device

        # Setup directories
        self.save_dir = Path(config["save_dir"])

        # Clean checkpoints directory if specified
        if config.get("clean_checkpoints", False):
            if self.save_dir.exists():
                print(f"Cleaning checkpoints directory: {self.save_dir}")
                shutil.rmtree(self.save_dir)

        # Create checkpoints directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize model
        self.model = create_medical_foundation_model(
            model_size=config["model_config"].get("model_size", "base"),
            img_size=config["model_config"].get("img_size", 256),
            in_chans=config["model_config"].get("in_channels", 3),
            patch_size=config["model_config"].get("patch_size", 16),
            drop_rate=config["model_config"].get("drop_rate", 0.1),
            attn_drop_rate=config["model_config"].get("attn_drop_rate", 0.1),
            drop_path_rate=config["model_config"].get("drop_path_rate", 0.1),
            use_llm=config["model_config"].get("use_llm", False),
        )

        self.model.to(device)
        print(f"Created KAN-based foundation model")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create losses
        self.setup_losses()

        # Initialize optimizer
        self.setup_optimizer()

        # Load dataset configurations
        self.setup_datasets()

        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()

        # Initialize visualizer
        self.visualizer = Visualizer(self.save_dir, self.logger)

        # Initialize report generator
        self.report_generator = ReportGenerator(self.metrics_collector, self.save_dir)

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def setup_logging(self):
        """Setup logging"""
        log_file = self.save_dir / "training.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def setup_losses(self):
        """Setup task-specific loss functions"""
        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()

        # Segmentation losses
        seg_loss_config = self.config.get("loss_config", {})

        # Create class weights to address class imbalance (background vs foreground)
        # This will put more weight on the foreground (positive) pixels
        class_weights = torch.tensor(
            [0.1, 0.9]
        )  # Background weight: 0.1, Foreground weight: 0.9

        # Create combined segmentation loss with Dice and Focal components
        self.seg_loss = CombinedLoss(
            ce_weight=seg_loss_config.get("ce_weight", 0.5),
            dice_weight=seg_loss_config.get("dice_weight", 0.5),
            use_focal=True,
            focal_alpha=seg_loss_config.get("focal_alpha", 0.75),
            focal_gamma=seg_loss_config.get("focal_gamma", 2.0),
            use_tversky=True,
            tversky_alpha=seg_loss_config.get("tversky_alpha", 0.3),
            tversky_beta=seg_loss_config.get("tversky_beta", 0.7),
            class_weights=(
                class_weights.to(self.device)
                if torch.cuda.is_available()
                else class_weights
            ),
        )

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Separate parameters for backbone and task-specific heads
        backbone_params = []
        head_params = []
        llm_params = []

        for name, param in self.model.named_parameters():
            if any(
                x in name for x in ["classification_heads", "segmentation_decoders"]
            ):
                head_params.append(param)
            elif "llm" in name:
                llm_params.append(param)
            else:
                backbone_params.append(param)

        # Use different learning rates for different components
        self.optimizer = optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": self.config["training_config"].get("backbone_lr", 1e-4),
                },
                {
                    "params": head_params,
                    "lr": self.config["training_config"].get("head_lr", 3e-4),
                },
                {
                    "params": llm_params,
                    "lr": self.config["training_config"].get("llm_lr", 5e-5),
                },
            ],
            weight_decay=self.config["training_config"].get("weight_decay", 1e-5),
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["training_config"].get("epochs_per_dataset", 15),
            eta_min=self.config["training_config"].get("min_lr", 1e-6),
        )

    def setup_datasets(self):
        """Setup datasets for sequential training"""
        # Extract dataset configurations from config
        self.dataset_configs = []

        # Process all datasets from the config
        for dataset in self.config["datasets"]:
            if dataset.get("enabled", True):
                # Get global data_sampling parameter if dataset-specific fraction is not provided
                limit_fraction = dataset.get(
                    "limit_samples_fraction",
                    self.config["training_config"].get("data_sampling", 1.0),
                )

                dataset_config = {
                    "path": dataset["path"],
                    "task": dataset["task"],
                    "modality": dataset.get("modality", "Unknown"),
                    "dataset_id": dataset["name"],
                    "name": dataset["name"],
                    "limit_samples_fraction": limit_fraction,
                    "enabled": dataset.get("enabled", True),
                }
                # Use global train_split setting
                dataset_config["train_split"] = self.config["training_config"].get("train_split", 0.8)

                # Add task-specific configurations
                if dataset["task"] == "classification":
                    dataset_config.update(
                        {
                            "classes": dataset["classes"],
                            "num_classes": dataset["num_classes"],
                        }
                    )
                elif dataset["task"] == "segmentation":
                    dataset_config.update(
                        {
                            "mask_suffix": dataset.get("mask_suffix", "_mask"),
                            "num_classes": dataset.get("num_classes", 2),
                        }
                    )

                self.dataset_configs.append(dataset_config)

        self.logger.info(
            f"Found {len(self.dataset_configs)} datasets for foundation model training"
        )
        for i, config in enumerate(self.dataset_configs):
            self.logger.info(f"Dataset {i}: {config['dataset_id']} ({config['task']})")

    def train_dataset(self, dataset_idx):
        """Train on a specific dataset"""
        config = self.dataset_configs[dataset_idx]
        dataset_id = config["dataset_id"]
        task_type = config["task"]

        print(f"\nTraining on dataset: {dataset_id} ({task_type})")
        print(f"{'='*60}")

        # Create dataset with appropriate image size
        img_size = self.config["model_config"].get("img_size", 256)

        # Apply global data_sampling parameter if present and no specific limit is set
        if (
            "limit_samples_fraction" not in config
            and "data_sampling" in self.config["training_config"]
        ):
            config["limit_samples_fraction"] = self.config["training_config"][
                "data_sampling"
            ]

        # Create a MultiDatasetLoader with just this dataset
        multi_loader = MultiDatasetLoader(
            [config],
            batch_size=self.config["training_config"].get("batch_size", 16),
            num_workers=self.config["training_config"].get("num_workers", 4),
            img_size=img_size,
            pin_memory=True,
        )

        if not multi_loader.datasets:
            print(f"No data found for dataset {dataset_id}!")
            return False

        # Get data loaders for this dataset
        train_loader, val_loader = multi_loader.get_dataloader(
            0
        )  # First (and only) dataset

        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        # Training epochs
        num_epochs = self.config["training_config"].get("epochs_per_dataset", 15)

        # Create epoch progress bar
        epoch_pbar = tqdm(range(num_epochs), desc=f"Training {dataset_id}")

        # Track best metrics for this dataset
        best_metric = 0.0
        best_epoch = 0

        # Dictionary to store history for plotting
        history = defaultdict(list)
        # Setup CSV logging for training metrics
        csv_path = self.save_dir / f"{dataset_id}_training_log.csv"
        # Write header
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if task_type == "classification":
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "train_accuracy",
                        "val_accuracy",
                        "precision_macro",
                        "recall_macro",
                        "f1_macro",
                    ]
                )
            else:  # segmentation
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "train_dice",
                        "val_dice",
                        "train_iou",
                        "val_iou",
                        "precision",
                        "recall",
                        "f1",
                        "specificity",
                    ]
                )

        for epoch in epoch_pbar:
            self.current_epoch = epoch

            # Train for one epoch
            train_loss, train_metrics = train_epoch(
                self.model,
                train_loader,
                task_type,
                self.device,
                self.optimizer,
                compute_loss,
                compute_metrics,
            )
            # Validate on validation set
            val_loss, val_metrics = validate(
                self.model,
                val_loader,
                task_type,
                self.device,
                compute_loss,
                compute_metrics,
            )

            # Append metrics to CSV
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if task_type == "classification":
                    writer.writerow(
                        [
                            epoch + 1,
                            train_loss,
                            val_loss,
                            train_metrics.get("accuracy", 0),
                            val_metrics.get("accuracy", 0),
                            train_metrics.get("precision_macro", 0),
                            train_metrics.get("recall_macro", 0),
                            train_metrics.get("f1_macro", 0),
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            epoch + 1,
                            train_loss,
                            val_loss,
                            train_metrics.get("dice", 0),
                            val_metrics.get("dice", 0),
                            train_metrics.get("iou", 0),
                            val_metrics.get("iou", 0),
                            train_metrics.get("precision", 0),
                            val_metrics.get("recall", 0),
                            train_metrics.get("f1", 0),
                            train_metrics.get("specificity", 0),
                        ]
                    )

            # Update history for plotting
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Store task-specific metrics
            if task_type == "classification":
                train_acc = train_metrics.get("accuracy", 0)
                val_acc = val_metrics.get("accuracy", 0)
                history["train_accuracy"].append(train_acc)
                history["val_accuracy"].append(val_acc)
                print(f"\nAccuracy: Train: {train_acc:.4f}, Test: {val_acc:.4f}")
                current_metric = val_acc
            else:  # segmentation
                train_dice = train_metrics.get("dice", 0)
                val_dice = val_metrics.get("dice", 0)
                train_iou = train_metrics.get("iou", 0)
                val_iou = val_metrics.get("iou", 0)

                history["train_dice"].append(train_dice)
                history["val_dice"].append(val_dice)
                history["train_iou"].append(train_iou)
                history["val_iou"].append(val_iou)

                print(f"\nDice: Train: {train_dice:.4f}, Test: {val_dice:.4f}")
                print(f"IoU: Train: {train_iou:.4f}, Test: {val_iou:.4f}")
                current_metric = val_dice

            print(f"Loss: Train: {train_loss:.4f}, Test: {val_loss:.4f}")

            # Progress metrics suppressed: no postfix displayed

            # Update metrics collector
            self.metrics_collector.update_history(
                dataset_id, train_metrics, phase="train"
            )
            self.metrics_collector.update_history(dataset_id, val_metrics, phase="val")

            # Check if this is the best model
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                best_epoch = epoch + 1  # 1-indexed for reporting
                self.metrics_collector.update_best_metric(
                    dataset_id, best_metric, best_epoch
                )
                self.metrics_collector.store_detailed_metrics(dataset_id, val_metrics)

                # Save checkpoint for best model
                self.save_checkpoint(is_best=True)

        # Plot training history
        self.visualizer.plot_training_history(dataset_id, history, task_type)

        print(f"Finished training on {dataset_id}")
        print(f"Best validation metric: {best_metric:.4f} (Epoch {best_epoch})")

        return True

    def train_all(self):
        """Train on all datasets sequentially"""

        for dataset_idx, dataset_config in enumerate(self.dataset_configs):
            if dataset_config.get("enabled", True):
                print(
                    f"\nTraining on dataset {dataset_idx+1}/{len(self.dataset_configs)}: {dataset_config['dataset_id']}"
                )
                success = self.train_dataset(dataset_idx)

                if not success:
                    print(
                        f"Warning: Training on dataset {dataset_config['dataset_id']} failed!"
                    )

                # Get test loader for final evaluation and visualization
                img_size = self.config["model_config"].get("img_size", 256)
                multi_loader = MultiDatasetLoader(
                    [dataset_config],
                    batch_size=self.config["training_config"].get("batch_size", 16),
                    num_workers=self.config["training_config"].get("num_workers", 4),
                    img_size=img_size,
                    pin_memory=True,
                )
                _, test_loader = multi_loader.get_dataloader(0)

                # Visualize results for segmentation datasets
                if dataset_config["task"] == "segmentation":
                    self.visualizer.visualize_segmentation_results(
                        dataset_config["dataset_id"],
                        self.model,
                        test_loader,
                        self.device,
                    )

        # Generate consolidated report
        self.report_generator.generate_consolidated_report()

        print("Training completed on all datasets!")

        # Return the metrics collector for external analysis
        return self.metrics_collector

    def finetune_all(self):
        """Fine-tune selected datasets based on config.fin etune settings"""
        # Load finetune settings
        ft_cfg = self.config.get("finetune", {})
        ft_datasets = ft_cfg.get("datasets", [])
        if not ft_datasets:
            print("No finetune datasets specified in config.")
            return False
        # Filter datasets for finetuning
        self.dataset_configs = [
            dc for dc in self.dataset_configs if dc["dataset_id"] in ft_datasets
        ]
        # Override training epochs and save_dir for finetune
        ft_epochs = ft_cfg.get("epochs_per_dataset", self.config["training_config"].get("epochs_per_dataset", 0))
        self.config["training_config"]["epochs_per_dataset"] = ft_epochs
        ft_save = Path(ft_cfg.get("save_dir", self.save_dir))
        # Update save directory for finetuning
        self.save_dir = ft_save
        # Re-init visualizer and report generator with new save_dir
        self.visualizer.save_dir = ft_save
        self.visualizer.fig_dir = ft_save / "figures"
        self.visualizer.vis_dir = ft_save / "visualizations"
        self.report_generator.save_dir = ft_save
        # Ensure directories exist
        ft_save.mkdir(parents=True, exist_ok=True)
        (ft_save / "figures").mkdir(parents=True, exist_ok=True)
        (ft_save / "visualizations").mkdir(parents=True, exist_ok=True)

        print(
            f"Loading pretrained checkpoint for fine-tuning: {ft_cfg.get('checkpoint')}"
        )
        # Load pretrained weights
        if not self.load_checkpoint(ft_cfg.get("checkpoint")):
            print("Error: Failed to load checkpoint for fine-tuning.")
            return False
        print(f"Starting fine-tuning on datasets: {ft_datasets}")
        success = True
        for idx, _ in enumerate(self.dataset_configs):
            res = self.train_dataset(idx)
            if not res:
                print(
                    f"Warning: Finetuning failed for dataset {self.dataset_configs[idx]['dataset_id']}"
                )
                success = False
        # After finetuning, regenerate report
        self.report_generator.generate_consolidated_report()
        print("Fine-tuning completed!")
        return success

    def save_checkpoint(self, is_best=False, suffix=""):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metrics": self.metrics_collector.get_best_metrics(),
            "global_step": self.global_step,
            "config": self.config,
        }

        if is_best:
            best_path = (
                self.save_dir / f"model_best{('_'+suffix) if suffix else ''}.pth"
            )
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"Saved best model checkpoint to {best_path}"
            )

    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            # Try to find the latest checkpoint
            checkpoint_path = self.save_dir / "checkpoint.pth"
            if not checkpoint_path.exists():
                # Try to find a best model
                best_model_path = self.save_dir / "model_best.pth"
                if best_model_path.exists():
                    checkpoint_path = best_model_path
                else:
                    self.logger.warning("No checkpoint found, starting from scratch")
                    return False

        if not Path(checkpoint_path).exists():
            self.logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return False

        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]

        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]

        if "best_metrics" in checkpoint:
            # Update metrics collector with best metrics
            for dataset_id, metrics in checkpoint["best_metrics"].items():
                self.metrics_collector.update_best_metric(
                    dataset_id, metrics["metric"], metrics.get("epoch", 0)
                )

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return True
