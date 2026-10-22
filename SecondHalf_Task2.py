import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=load_breast_cancer()
X=data.data
y=data.target
X_temp,X_test,y_temp,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)
X_train,X_val,y_train,y_val=train_test_split(
    X_temp,y_temp,test_size=0.25,
    random_state=42,stratify=y_temp
)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_val=scaler.transform(X_val)
X_test=scaler.transform(X_test)
X_train=torch.tensor(X_train,dtype=torch.float32)
X_val=torch.tensor(X_val,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)

y_train=torch.tensor(y_train,dtype=torch.float32).reshape(-1,1)
y_val=torch.tensor(y_val,dtype=torch.float32).reshape(-1,1)
y_test=torch.tensor(y_test,dtype=torch.float32).reshape(-1,1)
print("Traininig samples:",X_train.shape[0])
print("Validation samples:",X_val.shape[0])
print("Test samples:",X_test.shape[0])

class MLP(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_size,16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16,1)
        )

    def forward(self,x):
        return self.layers(x)

model=MLP(X_train.shape[1])
criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
train_losses=[]
val_accuracies=[]
best_val_loss=float("inf")
patience=8
waiting=0
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    output=model(X_train)
    loss=criterion(output,y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        val_output=model(X_val)
        val_loss=criterion(val_output,y_val)
        val_prediction=(val_output>=0).float()
        val_accuracy=(val_prediction==y_val).float().mean().item()
    val_accuracies.append(val_accuracy)
    if val_loss.item()<best_val_loss:
        best_val_loss=val_loss.item()
        waiting=0
        torch.save(model.state_dict(),"best_model.pth")
    else:
        waiting+=1
    print(f"Epoch{epoch+1}:"
              f"train_loss{loss.item():.4f},"
              f"val accuracy{val_accuracy:.4f}"
    )
    if waiting>=patience:
        print("Early stopping triggered!")
        break
model.load_state_dict(torch.load("best_model.pth",weights_only=True))
model.eval()
with torch.no_grad():
    test_output=model(X_test)
    test_prediction=(test_output>=0).float()
    test_accuracy=(test_prediction==y_test).float().mean().item()
print("\nPyTorch test accuracy:",round(test_accuracy,4))
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1,len(train_losses)+1),train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.subplot(1,2,2)
plt.plot(range(1,len(val_accuracies)+1),val_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.tight_layout()
plt.show()
sklearn_model=LogisticRegression(max_iter=2000)
sklearn_model.fit(X_train.numpy(),y_train.numpy().ravel())
sklearn_prediction=sklearn_model.predict(X_test.numpy())
sklearn_accuracy=accuracy_score(
    y_test.numpy().ravel(),sklearn_prediction
)
print("\nModel Comparison")
print("PyTorch test acccuracy:",round(test_accuracy,4))
print("Sklearn test accuracy:",round(sklearn_accuracy,4))