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
