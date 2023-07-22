
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train_tensor = torch.as_tensor(X_train).float().to(device)
X_val_tensor = torch.as_tensor(X_val).float().to(device)
