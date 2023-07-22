
X_tensor = torch.as_tensor(X).float()
y_tensor = torch.as_tensor(y).float()

train_ratio = 0.8

dataset = TensorDataset(X_tensor, y_tensor)

n_total = len(dataset)

n_train = int(n_total * train_ratio)

n_valid = n_total - n_train

train_data, valid_data = random_split(dataset, [n_train, n_valid])

train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

valid_loader = DataLoader(dataset=valid_data, batch_size=16)
