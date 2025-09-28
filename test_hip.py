# Create the comprehensive test file
import torch
import os

print("=== Comprehensive PyTorch/ROCm Test ===")
print(f"PyTorch version: {torch.__version__}")

# Check environment variables
print(f"\nEnvironment Variables:")
print(f"ROCM_PATH: {os.environ.get('ROCM_PATH', 'Not set')}")
print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not set')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')[:100]}...")

# Check CUDA availability
print(f"\nCUDA Status:")
print(f"CUDA available: {torch.cuda.is_available()}")

# Try to get more error information
try:
    torch.cuda.init()
    print("CUDA initialization successful")
except Exception as e:
    print(f"CUDA initialization failed: {e}")

# Check device count
try:
    device_count = torch.cuda.device_count()
    print(f"Device count: {device_count}")
    
    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Major:Minor: {props.major}.{props.minor}")
        
except Exception as e:
    print(f"Device enumeration failed: {e}")

# Test actual computation if devices available
if torch.cuda.is_available():
    print("\nTesting GPU computation...")
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print("✓ GPU matrix multiplication successful")
        print(f"Result shape: {y.shape}")
        print(f"Result device: {y.device}")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
        
        # Try with specific device
        try:
            x = torch.randn(1000, 1000).to('cuda:0')
            y = torch.matmul(x, x)
            print("✓ GPU computation successful with explicit device")
        except Exception as e2:
            print(f"✗ Explicit device also failed: {e2}")
else:
    print("\n❌ No GPU support available")
    
    # Try to get detailed error info
    try:
        # This might give us more specific error information
        test_tensor = torch.randn(10, 10)
        gpu_tensor = test_tensor.cuda()
        print("Unexpected: CUDA tensor creation succeeded")
    except Exception as e:
        print(f"Detailed error info: {e}")

