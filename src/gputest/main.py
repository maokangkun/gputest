import os
import sys
import pwd
import torch
import psutil
import platform
import datetime
import transformers
from torch.utils import benchmark
import pynvml # pip install nvidia-ml-py
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from rich import print as rprint

def print_title(title):
    n = len(title) + 2
    print('┌' + '─'*n + '┐')
    print(f'│ {title} │')
    print('└' + '─'*n + '┘')

def print_info(item, tab=''):
    if type(item[1]) is dict:
        rprint(f'[green]{item[0]}:[/]')
        t = ' '*4
        for i in item[1].items(): print_info(i, tab=tab+t)
    else:
        rprint(f'{tab}[green]{item[0]}:[/] [cyan]{item[1]}[/]')

def bytes2str(item):
    if type(item) is bytes:
        return item.decode()
    else:
        return item

def model_num_format(n):
    if n >= 1e12: # trillion
        return f'{n/1e12:.1f}T'
    elif n >= 1e9: # billion
        return f'{n/1e9:.1f}B'
    elif n >= 1e6: # million
        return f'{n/1e6:.1f}M'
    else:
        return f'{n/1000:.1f}K'

def size_num_format(n):
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}YB"

def watermark():
    print_title('Experimental Environment')
    if sys.platform == 'linux':
        os.system('lsb_release -a')
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
    elif sys.platform == 'darwin':
        device = 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built()  else 'cpu'
    elif sys.platform == 'win32':
        device = 'gpu' if torch.cuda.is_available() else 'cpu'

    user_info = pwd.getpwuid(os.getuid())
    base_info = {
        'platform': platform.platform(),
        'node': platform.node(),
        'time': datetime.datetime.now(),
        'user': user_info.pw_name,
        'shell': user_info.pw_shell,
        'current dir': os.getcwd(),
        'cpu': f'\\[logical] {psutil.cpu_count()}, \\[physical] {psutil.cpu_count(logical=False)}, \\[usage] {psutil.cpu_percent()}%',
        'virtual memory': f'\\[total] {size_num_format(psutil.virtual_memory().total)}, \\[avail] {size_num_format(psutil.virtual_memory().available)}, \\[used] {size_num_format(psutil.virtual_memory().used)} {psutil.virtual_memory().percent}%',
        'disk usage': f'\\[total] {size_num_format(psutil.disk_usage("/").total)}, \\[free] {size_num_format(psutil.disk_usage("/").free)}, \\[used] {size_num_format(psutil.disk_usage("/").used)} {psutil.disk_usage("/").percent}%',
    }
    gpu_info = {
        'device': device,
    }

    if device == 'gpu':
        pynvml.nvmlInit()
        print(f'CUDA version: {torch.version.cuda}')
        print(f'driver version: {bytes2str(pynvml.nvmlSystemGetDriverVersion())}')
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
        print(f'nccl version: {".".join(map(str,torch.cuda.nccl.version()))}')
        print(f'gpu usable count: {torch.cuda.device_count()}')
        deviceCount = pynvml.nvmlDeviceGetCount()
        print(f'gpu total count: {deviceCount}')
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = bytes2str(pynvml.nvmlDeviceGetName(handle))
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memstr = f'{int(meminfo.used/1024/1024):5d}M / {int(meminfo.total/1024/1024):5d}M, {meminfo.used/meminfo.total:3.0%}'
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            # fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            power_status = pynvml.nvmlDeviceGetPowerState(handle)
            print(f'    gpu {i}: {name}, [mem] {memstr}, {temp:3d}°C, 🔋 {power_status}')
        pynvml.nvmlShutdown()
        # print(torch.cuda.memory_summary())

        print('gpu direct communication matrix:')
        ret = os.popen('nvidia-smi topo -m')
        print(ret.read().split('\n\nLegend')[0])
        """
        X    = Self
        SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
        NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
        PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
        PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
        PIX  = Connection traversing at most a single PCIe bridge
        NV#  = Connection traversing a bonded set of # NVLinks
        在这里面, GPU间的通讯速度: NV# > PIX > PXB > PHB > NODE > SYS
        """
    elif device == 'mps':
        ...

    python_info = {
        'python interpreter': sys.executable,
        'python version': sys.version.replace('\n',''),
        'python packages version': {
            'torch': torch.__version__,
            'transformers': transformers.__version__,
        }
    }

    deepspeed_isinstalled = False
    try:
        import deepspeed
        python_info['python packages version'] |= {'deepspeed': deepspeed.__version__}
        deepspeed_isinstalled = True
    except ImportError: pass
    try:
        import flash_attn
        python_info['python packages version'] |= {'flash-attn': flash_attn.__version__}
    except ImportError: pass
    try:
        import triton
        python_info['python packages version'] |= {'triton': triton.__version__}
    except ImportError: pass

    info_dict = base_info | gpu_info | python_info
    list(map(print_info, info_dict.items()))

    if deepspeed_isinstalled:
        from deepspeed.env_report import cli_main
        print_title('DeepSpeed Report')
        cli_main()
    
    return device

def run_benchmark(device):
    print_title('Matrix Multiplication Benchmark')

    n = 1024 * 16
    op = 'A @ B'
    run_times = 50

    print(f'Matrix: A [{n} x {n}], B [{n} x {n}]')
    print(f'Operation: {op}')
    print(f'Experiment: {run_times}')
    print(f'Tensor:')

    for typ in [torch.float16, torch.float32]:
        a = torch.randn(n, n).type(typ).to(device)
        b = torch.randn(n, n).type(typ).to(device)

        bench = benchmark.Timer(
            stmt=op,
            globals={'A': a, 'B': b}
        )
        t = bench.timeit(run_times).median
        flops = 2*n**3 / t

        print(f'    - {typ} | {t:.5f}s (median) | {flops / 1e12:.4f} TFLOPS | GPU mem allocated {size_num_format(torch.cuda.max_memory_allocated())}, reserved {size_num_format(torch.cuda.max_memory_reserved())}')
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

def run_profiler(device):
    print_title('Resnet18 Inference Profiler')
    model = models.resnet18().to(device)
    inputs = torch.randn(5, 3, 224, 224).to(device)

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda': activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by=f"{device.type}_time_total", row_limit=10))
    prof.export_chrome_trace("resnet18_trace.json") # chrome://tracing/

def run_gpu_p2p_benchmark():
    """
    https://gist.github.com/joshlk/bbb1aca6e70b11d251886baee6423dcb

    1. Download repo git clone https://github.com/NVIDIA/cuda-samples.git
    2. You might need to install some additional packages sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev
    3. Either build everything by just execting make in root dir. Or cd Samples/5_Domain_Specific/p2pBandwidthLatencyTest/
    4. Edit Makefile: delete 89 90 from SMS & set CUDA_PATH to your conda env path (this conda env must install cudatoolkit-dev)
    5. Exectue: cd cuda-samples/bin/x86_64/linux/release; ./p2pBandwidthLatencyTest
    """
    print_title('P2P (Peer-to-Peer) GPU Bandwidth Latency Test')
    ret = os.popen('p2pBandwidthLatencyTest').read()
    i = ret.index('Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)')
    j = ret.index('Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)')
    print(ret[i:j])

def main():
    device = watermark()
    if device != 'cpu':
        run_benchmark(device)
        run_profiler(device)
    # else:
    #     print(f'[Warning] torch cuda: {torch.cuda.is_available()}, GPU total nums: {pynvml.nvmlDeviceGetCount()}, availabel: {torch.cuda.device_count()}')

if __name__ == '__main__':
    main()
