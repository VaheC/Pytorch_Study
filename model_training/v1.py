
epochs = 1000

losses = []

for _ in range(epochs):

    loss = train_step_fn(X_train_tensor, y_train_tensor)

    losses.append(loss)
