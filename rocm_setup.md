# AMD ROCm Setup Guide for Demucs GPU Acceleration

This guide summarizes the complete process to set up ROCm with PyTorch for Demucs GPU acceleration on AMD GPUs.

## Problem Statement

Demucs uses PyTorch for deep learning-based audio source separation. By default, it runs on CPU, but can achieve significant speed improvements (2.5x or more) when configured to use AMD GPUs via ROCm.

## Prerequisites

- AMD GPU with ROCm support (RDNA2/RDNA3 architecture recommended)
- Linux system (Ubuntu/Debian based)
- Python environment with Demucs installed

## Complete Setup Process

### Step 1: Install ROCm

```bash
# Add AMD ROCm repository (follow official AMD documentation for your Ubuntu version)
# Example for Ubuntu 22.04:
wget https://repo.radeon.com/amdgpu-install/6.2.2/ubuntu/jammy/amdgpu-install_6.2.60202-1_all.deb
sudo apt install ./amdgpu-install_6.2.60202-1_all.deb

# Install ROCm
sudo apt update
sudo apt install rocm-hip-sdk rocm-libs rocm-dev

# Verify ROCm installation
rocm-smi
cat /opt/rocm*/.info/version
```

### Step 2: Install ROCm-enabled PyTorch

```bash
# First, uninstall any CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install ROCm-enabled PyTorch (match your ROCm version)
# For ROCm 6.2:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify PyTorch installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
# Should show something like: 2.5.1+rocm6.2
```

### Step 3: Set Environment Variables

```bash
# Set ROCm path
export ROCM_PATH=/opt/rocm-6.2.2

# Set GFX version (critical for GPU detection)
# For RDNA2 GPUs (RX 6000 series, Radeon 6000M series):
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Set library paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-6.2.2/lib:/opt/rocm-6.2.2/lib64:/opt/rocm-6.2.2/hip/lib

# Add to bashrc for permanent settings
echo 'export ROCM_PATH=/opt/rocm-6.2.2' >> ~/.bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-6.2.2/lib:/opt/rocm-6.2.2/lib64:/opt/rocm-6.2.2/hip/lib' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Configure User Permissions

```bash
# Add user to required groups
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Verify group membership
groups $USER

# IMPORTANT: Log out and log back in, or reboot for group changes to take effect
```

### Step 5: Verify GPU Detection

```bash
# Test HIP detection
cat > test_hip_compile.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);

    if (err != hipSuccess) {
        std::cout << "HIP Error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "HIP Device Count: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);
        std::cout << "Device " << i << ": " << props.name << std::endl;
    }

    return 0;
}
EOF

# Compile and run
hipcc test_hip_compile.cpp -o test_hip_compile
./test_hip_compile

# Check ROCm system info
/opt/rocm-6.2.2/bin/rocminfo
```

### Step 6: Test PyTorch GPU Detection

```bash
# Create comprehensive test
cat > test_pytorch_gpu.py << 'EOF'
import torch
import os

print("=== PyTorch/ROCm GPU Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    # Test computation
    print("\nTesting GPU computation...")
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print("✓ GPU matrix multiplication successful")
        print(f"Result shape: {y.shape}")
        print(f"Result device: {y.device}")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
else:
    print("❌ No GPU support available")
EOF

# Run the test
python test_pytorch_gpu.py
```

### Step 7: Test Demucs GPU Usage

```bash
# Test Demucs with GPU monitoring
# In one terminal:
python -m demucs.separate -v your_audio_file.mp3

# In another terminal, monitor GPU usage:
watch -n 1 rocm-smi

# Or for more detailed monitoring:
rocm-smi --showuse --showpower
```

## Troubleshooting Common Issues

### Issue 1: Version Mismatch

**Problem**: PyTorch version doesn't match ROCm version
**Solution**: Ensure PyTorch ROCm version matches your installed ROCm version

```bash
# Check ROCm version
cat /opt/rocm*/.info/version

# Install matching PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2  # Adjust version
```

### Issue 2: Wrong GFX Version

**Problem**: `HSA_OVERRIDE_GFX_VERSION` set incorrectly
**Solution**: Find correct GFX version for your GPU

```bash
# Check GPU GFX version
rocm-smi --showproductname

# Try different GFX versions
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # RDNA2
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RDNA3
export HSA_OVERRIDE_GFX_VERSION=9.0.0   # GCN5
```

### Issue 3: Permission Issues

**Problem**: User not in required groups
**Solution**: Add user to video and render groups, then restart

```bash
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER
# REBOOT REQUIRED
```

### Issue 4: Environment Variables Not Set

**Problem**: ROCm environment variables not properly configured
**Solution**: Set all required environment variables and add to bashrc

```bash
export ROCM_PATH=/opt/rocm-6.2.2
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-6.2.2/lib:/opt/rocm-6.2.2/lib64:/opt/rocm-6.2.2/hip/lib
```

### Issue 5: Incomplete ROCm Installation

**Problem**: Missing ROCm components
**Solution**: Reinstall ROCm completely

```bash
sudo apt remove --purge rocm-hip-sdk rocm-libs rocm-dev
sudo apt autoremove
sudo rm -rf /opt/rocm*
# Reinstall following official AMD documentation
```

## Performance Monitoring

### Monitor GPU Usage During Demucs Processing

```bash
# Real-time monitoring
watch -n 1 rocm-smi

# Detailed monitoring
rocm-smi --showuse --showpower --showtemp

# Continuous logging
rocm-smi --showuse --showpower --loop
```

### Compare CPU vs GPU Performance

```bash
# Time CPU processing
time python -m demucs.separate -d cpu your_audio_file.mp3

# Time GPU processing
time python -m demucs.separate -d cuda your_audio_file.mp3
```

## Verification Checklist

After completing the setup, verify:

- [ ] `rocm-smi` shows both GPUs
- [ ] `rocminfo` shows HSA agents
- [ ] HIP test compiles and runs successfully
- [ ] PyTorch detects GPUs (`torch.cuda.is_available()` returns True)
- [ ] PyTorch GPU computation test passes
- [ ] Demucs runs significantly faster on GPU (2x+ improvement)
- [ ] `rocm-smi` shows GPU usage during Demucs processing

## Expected Results

When properly configured, you should see:

- **HIP Device Count: 2** (or more depending on your setup)
- **CUDA available: True** in PyTorch
- **Device count: 2** in PyTorch
- **2.5x or more speed improvement** in Demucs processing
- **GPU usage spikes** in `rocm-smi` during audio separation

## Final Notes

- **Restart is crucial**: Many changes (especially group permissions) require a full restart
- **Version matching is critical**: Ensure PyTorch ROCm version matches your ROCm installation
- **Environment variables matter**: `HSA_OVERRIDE_GFX_VERSION` is often the key to success
- **Monitor performance**: Use `rocm-smi` to verify GPU is actually being used

This setup should provide reliable GPU acceleration for Demucs on AMD GPUs using ROCm.
