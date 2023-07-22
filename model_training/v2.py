
epochs = 1000

losses = []

for _ in range(epochs):

    mini_batch_losses = []

    for x_batch, y_batch in train_loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        mini_batch_losses.append(train_step_fn(x_batch, y_batch))

    loss = np.mean(mini_batch_losses)

    losses.append(loss)
