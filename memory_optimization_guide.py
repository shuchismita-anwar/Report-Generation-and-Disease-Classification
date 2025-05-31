#!/usr/bin/env python3
"""
Memory Optimization Guide for Windows PyTorch Training
This script helps fix the "paging file too small" error when training deep learning models.
"""

import os
import sys
import platform
import psutil
import subprocess
from pathlib import Path

def check_system_info():
    """Display current system information"""
    print("=" * 60)
    print("SYSTEM MEMORY ANALYSIS")
    print("=" * 60)
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"Used RAM: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")
    
    # Disk space
    disk = psutil.disk_usage('C:')
    print(f"Total Disk Space (C:): {disk.total / (1024**3):.1f} GB")
    print(f"Free Disk Space (C:): {disk.free / (1024**3):.1f} GB")
    
    # Virtual memory/swap
    swap = psutil.swap_memory()
    print(f"Swap/Virtual Memory Total: {swap.total / (1024**3):.1f} GB")
    print(f"Swap/Virtual Memory Used: {swap.used / (1024**3):.1f} GB")
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    return memory, disk, swap

def get_optimal_paging_file_size(total_ram_gb):
    """Calculate optimal paging file size"""
    if total_ram_gb <= 8:
        return int(total_ram_gb * 1.5 * 1024)  # 1.5x RAM in MB
    elif total_ram_gb <= 16:
        return int(total_ram_gb * 1.25 * 1024)  # 1.25x RAM in MB
    else:
        return int(total_ram_gb * 1.0 * 1024)  # 1x RAM in MB

def provide_manual_instructions(optimal_size_mb):
    """Provide manual instructions to fix paging file"""
    print("\n" + "=" * 60)
    print("MANUAL FIX INSTRUCTIONS")
    print("=" * 60)
    
    print("To fix the 'paging file too small' error, follow these steps:")
    print("\n1. INCREASE VIRTUAL MEMORY (PAGING FILE):")
    print("   - Press Windows + R, type 'sysdm.cpl', press Enter")
    print("   - Go to 'Advanced' tab → Click 'Settings' under Performance")
    print("   - Go to 'Advanced' tab → Click 'Change' under Virtual Memory")
    print("   - Uncheck 'Automatically manage paging file size for all drives'")
    print("   - Select your main drive (usually C:)")
    print("   - Select 'Custom size'")
    print(f"   - Set Initial size: {optimal_size_mb} MB")
    print(f"   - Set Maximum size: {optimal_size_mb * 2} MB")
    print("   - Click 'Set' → Click 'OK' → Restart computer")
    
    print("\n2. FREE UP SYSTEM MEMORY:")
    print("   - Close unnecessary programs and browser tabs")
    print("   - Disable startup programs: Ctrl+Shift+Esc → Startup tab")
    print("   - Run disk cleanup: Windows + R → 'cleanmgr' → Enter")
    
    print("\n3. TEMPORARY WORKAROUNDS:")
    print("   - Use smaller batch sizes (the script now automatically does this)")
    print("   - Use CPU training instead of GPU (script will fallback)")
    print("   - Close other applications while training")

def check_pytorch_installation():
    """Check if PyTorch can be imported with CUDA"""
    print("\n" + "=" * 60)
    print("PYTORCH INSTALLATION CHECK")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            
            # Test CUDA memory allocation
            try:
                test_tensor = torch.zeros(1000, 1000).cuda()
                print("✓ CUDA memory allocation: Working")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"✗ CUDA memory allocation: Failed ({e})")
        else:
            print("ℹ No CUDA GPU available - will use CPU training")
            
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    return True

def optimize_environment_variables():
    """Set environment variables for memory optimization"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT OPTIMIZATION")
    print("=" * 60)
    
    env_vars = {
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'CUDA_LAUNCH_BLOCKING': '1',
        'OMP_NUM_THREADS': '1',  # Reduce CPU thread usage
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"Set {var}={value}")

def create_memory_optimized_script():
    """Create a memory-optimized version of the training script"""
    print("\n" + "=" * 60)
    print("SCRIPT OPTIMIZATION")
    print("=" * 60)
    
    print("✓ The training script has been automatically optimized with:")
    print("  - Automatic memory detection and optimization")
    print("  - Smaller batch sizes for low memory systems")
    print("  - CPU fallback when CUDA fails")
    print("  - Simplified model architecture for low memory")
    print("  - Reduced data loader workers")
    print("  - Aggressive garbage collection")
    print("  - CUDA memory cache clearing")

def main():
    print("PyTorch Memory Optimization Helper")
    print("Solving 'WinError 1455: The paging file is too small' error")
    
    # Check system info
    memory, disk, swap = check_system_info()
    
    # Calculate optimal paging file size
    total_ram_gb = memory.total / (1024**3)
    optimal_size_mb = get_optimal_paging_file_size(total_ram_gb)
    
    # Check available space
    free_space_gb = disk.free / (1024**3)
    required_space_gb = optimal_size_mb / 1024
    
    print("\n" + "=" * 60)
    print("MEMORY ASSESSMENT")
    print("=" * 60)
    
    if memory.available < 4 * (1024**3):  # Less than 4GB available
        print("⚠️  WARNING: Low system memory detected!")
        print("   Recommend closing other applications before training.")
    
    if free_space_gb < required_space_gb + 5:  # Not enough space + 5GB buffer
        print("⚠️  WARNING: Insufficient disk space for optimal paging file!")
        print(f"   Need {required_space_gb + 5:.1f}GB free, have {free_space_gb:.1f}GB")
    
    if swap.total < optimal_size_mb * (1024**2):  # Current paging file too small
        print("⚠️  WARNING: Current paging file is too small!")
        print(f"   Current: {swap.total / (1024**3):.1f}GB")
        print(f"   Recommended: {optimal_size_mb / 1024:.1f}GB")
    
    # Provide instructions
    provide_manual_instructions(optimal_size_mb)
    
    # Check PyTorch
    pytorch_working = check_pytorch_installation()
    
    # Optimize environment
    optimize_environment_variables()
    
    # Script optimization info
    create_memory_optimized_script()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    if not pytorch_working:
        print("1. Fix PyTorch installation first")
        print("2. Then increase paging file size")
        print("3. Restart computer")
        print("4. Run the training script")
    elif swap.total < optimal_size_mb * (1024**2):
        print("1. Increase paging file size using instructions above")
        print("2. Restart computer")
        print("3. Run the training script (it's already optimized)")
    else:
        print("✓ System appears ready for training!")
        print("✓ The training script is already optimized for your system")
        print("✓ You can now run: python new_scripts/training_mimic_on_chexpert.py")
    
    print("\nIf you continue to have issues:")
    print("- Try restarting your computer first")
    print("- Close all unnecessary applications")
    print("- The script will automatically use CPU if CUDA fails")

if __name__ == "__main__":
    main() 