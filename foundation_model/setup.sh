#!/bin/bash

# Medical Image Analysis Foundation Model Setup Script
echo "🏥 Setting up Medical Image Analysis Foundation Model Environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust URL based on your CUDA version)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📋 Installing other dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Setting up directories..."
python3 utils.py --config config.yaml --action setup

# Validate dataset structure
echo "🔍 Validating dataset structure..."
python3 utils.py --config config.yaml --action validate

# Check if datasets are valid
if [ $? -eq 0 ]; then
    echo "✅ Dataset validation passed!"
    
    # Create dataset info
    echo "📊 Creating dataset information..."
    python3 utils.py --config config.yaml --action info
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📚 Quick Start Guide:"
    echo "==================="
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Start training:"
    echo "   python train.py --config config.yaml"
    echo ""
    echo "3. Run inference on a new image:"
    echo "   python inference.py --model_path models/foundation_model_final.pth \\"
    echo "                       --image_path path/to/image.jpg \\"
    echo "                       --task classification \\"
    echo "                       --dataset_name brain_mri_nd5 \\"
    echo "                       --visualize"
    echo ""
    echo "4. Evaluate the trained model:"
    echo "   python evaluate.py --model_path models/foundation_model_final.pth"
    echo ""
    echo "📖 For more information, check README.md"
    
else
    echo "❌ Dataset validation failed!"
    echo "Please check your dataset structure and fix any issues before training."
    echo ""
    echo "Expected structure:"
    echo "data/"
    echo "├── classification/"
    echo "│   ├── brain_mri_nd5/     # Brain MRI classification"
    echo "│   ├── hf_brain_tumor/    # Brain tumor classification"
    echo "│   └── pancreas/          # Pancreas classification"
    echo "└── segmentation/"
    echo "    ├── btcv/              # Multi-organ segmentation"
    echo "    ├── cvc_clinicdb/      # Polyp segmentation"
    echo "    ├── kvasir_seg/        # GI tract segmentation"
    echo "    ├── medseg_covid/      # COVID lung segmentation"
    echo "    └── medseg_liver/      # Liver segmentation"
    echo ""
    echo "Run 'python utils.py --config config.yaml --action validate' for detailed error information."
fi

echo ""
echo "🔧 Environment setup completed!"
echo "Virtual environment: $(pwd)/venv"
