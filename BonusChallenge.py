import torch
import time

size=1000
a=torch.randn(size,size)
b=torch.randn(size,size)
start=time.time()
result=a@b
cpu_time=time.time()-start
print("CPU time:",cpu_time,"seconds")
if torch.cuda.is_available():
    a_gpu=a.to("cuda")
    b_gpu=b.to("cuda")
    torch.cuda.synchronize()
    start=time.time()
    result_gpu=a_gpu@b_gpu
    torch.cuda.synchronize()
    gpu_time=time.time()-start
    print("GPU time:",gpu_time,"seconds")
else:
    print("GPU is not available.Run on CPU or try Google Colab.")