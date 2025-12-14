#!/bin/bash
set -e

echo "=== Fraud Detection Setup ==="
echo ""

# Check Python version
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "  Python $PY_VERSION found"

    # Check if version >= 3.10
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
        echo "  Version OK"
    else
        echo "  ERROR: Python 3.10+ required, found $PY_VERSION"
        exit 1
    fi
else
    echo "  ERROR: Python3 not found"
    echo "  Install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

# Check/Install uv
echo ""
echo "Checking uv..."
if command -v uv &> /dev/null; then
    echo "  uv found: $(uv --version)"
else
    echo "  uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "  uv installed: $(uv --version)"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
uv sync
echo "  Done"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/raw data/processed models
echo "  Done"

# Check CUDA/GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  Found: $GPU_NAME"
    echo "  XGBoost will use GPU automatically"
else
    echo "  No NVIDIA GPU detected (will use CPU)"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run notebooks:"
echo "  cd notebooks && jupyter notebook"
echo ""
echo "Or activate venv manually:"
echo "  source .venv/bin/activate"
echo ""