import torch
import numpy as np

a=torch.tensor([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
])
b=torch.zeros(2,3)
c=torch.ones(2,3)
d=torch.arange(1,7)
e=torch.randn(2,3)
print("Tensor a:\n",a)
print("Zeros:\n",b)
print("Ones:\n",c)
print("Arange:\n",d)
print("Random Tensor:\n",e)
print("\n Shape:",a.shape)
print("Datatype",a.dtype)
print("Device:",a.device)
print("\n First row:",a[0])
print("First two row:\n",a[:,2])
print("First two Column\n",a[:,:2])
print("\nView\n",a.view(2,6))
print("\nReshape:\n",a.reshape(4,3))
print("\n Transpose:\n",a.T)
x=torch.arange(1,13).view(3,4)
y=torch.arange(1,9).view(4,2)

result1=x@y
result2=torch.matmul(x,y)
print("\nMatrix multiplication using @:\n",result1)
print("Result shape:",result1.shape)

print("\nMatrix multiplication using torch.matmul:\n",result2)
print("Result shape",result2.shape)

numpy_array=np.array([1,2,3,4],dtype=np.float32)
tensor_from_numpy=torch.from_numpy(numpy_array)
print("\nTensor from numpy:",tensor_from_numpy)
numpy_result=tensor_from_numpy.numpy()
print("Converted to Numpy:",numpy_result)
if torch.cuda.is_available():
    device=torch.device("cuda")
    print("\nGPU is available")
else:
    device=torch.device("cpu")
    print("\nGPU is not available.Using CPU")
a=a.to(device)
print("Final device:",a.device)