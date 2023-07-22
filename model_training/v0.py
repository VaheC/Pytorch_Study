
epochs = 1000

for _ in range(epochs):

    model.train()

    y_train_tensor_hat = model(X_train_tensor)

    loss = loss_fn(y_train_tensor_hat, y_train_tensor)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
