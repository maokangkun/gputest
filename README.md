# gputest
Reporting GPU benchmark results and information.

## Installation
Just `pip install gputest` !

## Usage

```bash
> gputest
```

example output:
```
Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.3 LTS
Release:	22.04
Codename:	jammy
┌──────────────────────────┐
│ Experimental Environment │
└──────────────────────────┘
platform: Linux-5.10.16.3-microsoft-standard-WSL2-x86_64-with-glibc2.35
node: DESKTOP-88GMTT4
time: 2023-10-09 12:44:20.726428
python interpreter: /home/kk/miniconda3/bin/python
python version: 3.11.4 (main, Jul  5 2023, 13:45:01) [GCC 11.2.0]
device: gpu
CUDA version: 12.1
driver version: 537.13
cuDNN version: 8902
nccl version: 2.18.1
gpu usable count: 1
gpu total count: 1
    gpu 0: NVIDIA GeForce RTX 4090, [mem]  1380M / 24564M,  6%,  31°C, 🔋 8
gpu direct communication matrix:
	    GPU0	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 				N/A
cpu: [logical] 24, [physical] 12, [usage] 3.6%
virtual memory: [total] 15.5GB, [avail] 14.2GB, [used] 1.0GB 8.7%
disk usage: [total] 251.0GB, [free] 228.9GB, [used] 9.3GB 3.9%
current dir: /home/kk
user: kk
shell: /bin/bash
python packages version:
    torch: 2.1.0+cu121
    transformers: 4.34.0
    triton: 2.1.0
┌─────────────────────────────────┐
│ Matrix Multiplication Benchmark │
└─────────────────────────────────┘
Matrix: A [16384 x 16384], B [16384 x 16384]
Operation: A @ B
Experiment: 50
Tensor:
    - torch.float16 | 0.05057s (median) | 173.9521 TFLOPS | GPU mem allocated 1.5GB, reserved 1.5GB
    - torch.float32 | 0.16625s (median) | 52.9086 TFLOPS | GPU mem allocated 3.0GB, reserved 4.5GB
┌─────────────────────────────┐
│ Resnet18 Inference Profiler │
└─────────────────────────────┘
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  model_inference         0.16%       1.312ms       100.00%     803.729ms     803.729ms           0 b           0 b           0 b    -110.74 Mb             1  
                     aten::conv2d         0.19%       1.565ms        91.57%     735.938ms      36.797ms           0 b           0 b      47.51 Mb     490.00 Kb            20  
                aten::convolution         0.02%     176.000us        91.56%     735.875ms      36.794ms           0 b           0 b      47.51 Mb           0 b            20  
               aten::_convolution         0.02%     143.000us        91.54%     735.699ms      36.785ms           0 b           0 b      47.51 Mb           0 b            20  
          aten::cudnn_convolution        91.52%     735.556ms        91.52%     735.556ms      36.778ms           0 b           0 b      47.51 Mb      47.51 Mb            20  
                       aten::add_         0.06%     467.000us         0.06%     467.000us      16.679us           0 b           0 b           0 b           0 b            28  
                 aten::batch_norm         0.01%      64.000us         7.91%      63.586ms       3.179ms           0 b           0 b      47.41 Mb       3.83 Mb            20  
     aten::_batch_norm_impl_index         0.01%      62.000us         7.91%      63.551ms       3.178ms           0 b           0 b      47.41 Mb           0 b            20  
           aten::cudnn_batch_norm         7.77%      62.451ms         7.90%      63.489ms       3.174ms           0 b           0 b      47.41 Mb       5.50 Kb            20  
                 aten::empty_like         0.01%      56.000us         0.12%     984.000us      49.200us           0 b           0 b      47.37 Mb           0 b            20  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 803.729ms

```


```python
from gputest.nvidia import get_gpus

for gpu in get_gpus():
	print(gpu.__dict__)
	print(gpu.get_max_clock_speeds())
	print(gpu.get_clock_speeds())
	print(gpu.get_memory_details())
```

```python
from gpuinfo.windows import get_gpus

for gpu in get_gpus():
	print(gpu.__dict__)
```
