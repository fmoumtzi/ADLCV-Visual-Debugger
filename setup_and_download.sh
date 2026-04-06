#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================="
echo "VisualDebugger - Environment & Data Setup"
echo "========================================="

# 1. Setup Conda Environment
echo "--> Initializing Miniforge..."
source ~/miniforge3/bin/activate

# check if conda env 'adlcv' exists
if ! conda env list | grep -q "^adlcv "; then
    echo "--> Creating Conda environment 'adlcv' with Python 3.10..."
    conda create -n adlcv python=3.10 -y
else
    echo "--> Conda environment 'adlcv' already exists."
fi

echo "--> Activating Conda environment 'adlcv'..."
conda activate adlcv

echo "--> Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup Directories
mkdir -p data/vqa
mkdir -p data/coco
mkdir -p models

# Download Models (using Hugging Face CLI)
# If downloading a gated model need to be logged in to huggingface-cli first
echo "--> Downloading Models to ./models directory..."
export HF_HOME="./models" # Forces models to download here instead of ~/.cache

# VLM
echo "Downloading Qwen2.5-VL-3B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/Qwen2.5-VL-3B-Instruct


# LLM as a judge
echo "Downloading Prometheus-2-7B..."
huggingface-cli download prometheus-eval/prometheus-7b-v2.0 --local-dir ./models/Prometheus-2-7B

# Patronus-Lynx Model
#echo "Downloading Patronus-Lynx-8B-v1.1..."
#huggingface-cli download PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-v1.1 --local-dir ./models/Patronus-Lynx-8B

# Datasets
echo "--> Downloading VQA v2 Annotations and Questions..."
cd data/vqa
# VQA v2 Training and Validation questions/annotations
wget -nc https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget -nc https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget -nc https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget -nc https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

echo "Unzipping VQA data..."
unzip -q -n '*.zip'
rm *.zip
cd ../..

echo "--> Downloading COCO 2014 Images (Required for VQA v2)..."
cd data/coco
# VQA v2 relies on the COCO 2014 image splits (13GB + 6GB)
wget -nc http://images.cocodataset.org/zips/train2014.zip
wget -nc http://images.cocodataset.org/zips/val2014.zip

echo "Unzipping COCO images (this will take a while)..."
unzip -q -n train2014.zip
unzip -q -n val2014.zip
rm *.zip
cd ../..

echo "========================================="
echo "Setup Complete! "
echo "Run 'source venv/bin/activate' to start working."
echo "========================================="