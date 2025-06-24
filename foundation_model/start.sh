#!/bin/bash

# filepath: /home/jobayer/research/mhs/foundation_model/start.sh
# This script launches the medical foundation model training
# Usage: ./start.sh [operation] [options]

# Set the operation (default: help) and shift it out from the arguments
OPERATION=${1:-help}
shift

# Set default configuration and device (these can be overridden by options)
CONFIG="config.json"
DEVICE="cuda"

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Process optional arguments
while [[ $# -gt 0 ]]; do            case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            EXTRA_ARGS="$EXTRA_ARGS --dataset $2"
            shift 2
            ;;
        --enable-datasets)
            EXTRA_ARGS="$EXTRA_ARGS --enable-datasets $2"
            shift 2
            ;;
        --disable-datasets)
            EXTRA_ARGS="$EXTRA_ARGS --disable-datasets $2"
            shift 2
            ;;
        --freeze-backbone)
            EXTRA_ARGS="$EXTRA_ARGS --freeze-backbone"
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            EXTRA_ARGS="$EXTRA_ARGS --checkpoint $2"
            shift 2
            ;;
        *)
            # Pass all other arguments directly to the script
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Ensure the config file exists
if [ ! -f "$CONFIG" ]; then
    if [ -f "kan_config.json" ] && [ "$CONFIG" = "config.json" ]; then
        # Use kan_config.json as fallback if the default config doesn't exist
        echo "Warning: config.json not found, using kan_config.json instead"
        CONFIG="kan_config.json"
    else
        echo "Error: Configuration file $CONFIG not found."
        exit 1
    fi
fi

# Execute the requested operation
case "$OPERATION" in
    train)
        echo "======================================="
        echo "Starting KAN Foundation Model Training"
        echo "======================================="
        echo "Using config: $CONFIG"
        echo "Using device: $DEVICE"
        
        PYTHONPATH="$PWD" python foundation_model.py --config "$CONFIG" --device "$DEVICE" $EXTRA_ARGS
        
        if [ $? -eq 0 ]; then
            echo "Training completed successfully!"
            echo "Checkpoints saved in directory specified in config file."
        else
            echo "Error: Training failed."
            exit 1
        fi
        ;;
        
    finetune)
        echo "========================================="
        echo "Fine-tuning KAN Foundation Model"
        echo "========================================="
        echo "Using config: $CONFIG"
        echo "Using device: $DEVICE"
        
        # Run finetuning with pretrained checkpoint
        PYTHONPATH="$PWD" python foundation_model.py --config "$CONFIG" --device "$DEVICE" --mode finetune --resume "$CHECKPOINT" $EXTRA_ARGS

        if [ $? -eq 0 ]; then
            echo "Fine-tuning completed successfully!"
            echo "Checkpoints saved in directory specified in config file."
        else
            echo "Error: Fine-tuning failed."
            exit 1
        fi
        ;;
        
    test)
        echo "========================================="
        echo "Testing KAN Foundation Model"
        echo "========================================="
        echo "Using config: $CONFIG_FILE"
        echo "Using device: $DEVICE"
        
        PYTHONPATH="$PWD" python foundation_model.py --config "$CONFIG_FILE" --device "$DEVICE" --mode test $EXTRA_ARGS
        
        if [ $? -eq 0 ]; then
            echo "Testing completed successfully!"
        else
            echo "Error: Testing failed."
            exit 1
        fi
        ;;
        
    help)
        echo "Medical Foundation Model - Usage Guide"
        echo "======================================"
        echo "Commands:"
        echo "  train    - Train a new foundation model"
        echo "  finetune - Fine-tune an existing model"
        echo "  test     - Run model on test datasets"
        echo "  help     - Show this help message"
        echo ""
        echo "Options:"
        echo "  --config FILE              - Path to configuration file (default: config.json)"
        echo "  --device DEVICE            - Device to use (cuda/cpu, default: auto-detect)"
        echo "  --dataset NAME             - Specific dataset to train/test on"
        echo "  --enable-datasets LIST     - Comma-separated list of datasets to enable"
        echo "  --disable-datasets LIST    - Comma-separated list of datasets to disable"
        echo "  --checkpoint PATH          - Path to checkpoint for resuming or finetuning"
        echo "  --freeze-backbone          - Freeze backbone layers during finetuning"
        echo ""
        echo "Examples:"
        echo "  ./start.sh train"
        echo "  ./start.sh train --config my_config.json"
        echo "  ./start.sh train --enable-datasets brain_mri_nd5,kvasir_seg"
        echo "  ./start.sh finetune --checkpoint checkpoints/epoch_10.pth"
        echo "  ./start.sh finetune --checkpoint checkpoints/best.pth --freeze-backbone"
        echo "  ./start.sh test --checkpoint checkpoints/best.pth --dataset kvasir_seg"
        echo "  ./start.sh test --config test_config.json --device cpu"
        ;;
        
    *)
        echo "Error: Unknown operation '$OPERATION'"
        echo "Run './start.sh help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "Operation completed."
