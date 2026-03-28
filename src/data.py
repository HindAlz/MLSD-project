from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def get_dataloaders(batch_size: int = 32):
    train=pd.read_csv("data/train.csv")
    test=pd.read_csv("data/test.csv")
    X_train = train.drop(columns=["fake"]).values
    y_train = train["fake"].values

    X_val = test.drop(columns=["fake"]).values
    y_val = test["fake"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    return train_loader, val_loader, input_dim, num_classes


get_dataloaders()