import torch
w=torch.tensor(1.0,requires_grad=True)
b=torch.tensor(0.0,requires_grad=True)
loss=(w*2+b-10)**2
loss.backward()
print("Initial loss:",loss.item())
print("w.grad=",w.grad.item())
print("b.grad=",b.grad.item())
w=torch.tensor(1.0,requires_grad=True)
b=torch.tensor(0.0,requires_grad=True)
learning_rate=0.01
for i in range(30):
    loss=(w*2+b-10)**2
    loss.backward()
    with torch.no_grad():
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad
    w.grad.zero_()
    b.grad.zero_()
print("\nAfter 30Steps")
print("w=",w.item())
print("b=",b.item())
print("Loss=",loss.item())

w=torch.tensor(1.0,requires_grad=True)
print("\nWithout zeroing gradient:")
for i in range(3):
    loss=(w*2-10)**2
    loss.backward()
    print("Step",i+1,"Gradient:",w.grad.item())
print("\nExplanation:")
print("require+grad=True tells PyTorch to track operations for gradients.")
print("backward() calculates gradients automatially")
print("detach() creates a tensor separated from the autograd graph.")
