## Restormer Deshadowing - One-Click Google Colab Runner
## Designed for NVIDIA Tesla T4 (15GB) & Google Colab environments

import os
import sys
import argparse
import subprocess

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

def check_environment():
    print("=" * 60)
    print("       RESTORMER DESHADOWING - COLAB ENVIRONMENT CHECK       ")
    print("=" * 60)
    
    import torch
    if not torch.cuda.is_available():
        print("❌ ERROR: No CUDA GPU detected!")
        print("Please enable GPU in Colab: Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"✅ GPU Detected : {gpu_name}")
    print(f"✅ Total VRAM   : {vram_total:.2f} GB")
    print(f"✅ PyTorch      : {torch.__version__}")
    print("=" * 60)

def verify_dataset(train_input_dir, train_target_dir):
    if not os.path.exists(train_input_dir) or len(os.listdir(train_input_dir)) == 0:
        print(f"⚠️ Warning: Train input directory is empty or missing: {train_input_dir}")
        print("Please copy or mount your dataset into Deshadowing/Datasets/train/ISTD/")
        return False
    
    num_inputs = len([f for f in os.listdir(train_input_dir) if not f.startswith('.')])
    num_targets = len([f for f in os.listdir(train_target_dir) if not f.startswith('.')])
    print(f"✅ Dataset found: {num_inputs} input images, {num_targets} target images.")
    return True

def run_train(config_file):
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "basicsr", "train.py"),
        "-opt", config_file
    ]
    print(f"\n🚀 Launching training with config: {config_file}")
    print(f"Command: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training exited with error code {e.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Colab Runner for Restormer Deshadowing")
    parser.add_argument("--config", default="Deshadowing/Options/Deshadowing_Restormer_Colab_T4.yml", type=str, help="Path to config yaml")
    parser.add_argument("--check_only", action="store_true", help="Only check GPU and dataset without starting training")
    args = parser.parse_args()

    check_environment()

    train_input = os.path.join(REPO_ROOT, "Deshadowing", "Datasets", "train", "ISTD", "input")
    train_target = os.path.join(REPO_ROOT, "Deshadowing", "Datasets", "train", "ISTD", "target")
    verify_dataset(train_input, train_target)

    if args.check_only:
        print("Check completed successfully.")
        return

    config_path = os.path.join(REPO_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    run_train(config_path)

if __name__ == "__main__":
    main()
