import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(30,16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16,1)
        )
    def forward(self,x):
        return self.layers(x)

model=MLP()
print("Model Summary")
print(model)
total_params=sum(p.numel() for p in model.parameters())
print("Total Parameter",total_params)
embedding=nn.Embedding(10,4)
indices=torch.tensor([1,3,5,7])
output=embedding(indices)
print("\nEmbedding output Shape:",output.shape)
batch_norm=nn.BatchNorm1d(16)
data=torch.randn(8,16)
normalized_data=batch_norm(data)
print("\n BatchNorm output shape",normalized_data.shape)
print("Mean per Feature:",normalized_data.mean(dim=0))
print("Std per feature:",normalized_data.std(dim=0,unbiased=False))
model.train()
print("\nModel is in training mode:",model.training)
model.eval()
print("Model is in evaluation mode:",model.training)