
epochs = 1000

losses = []

for _ in range(epochs):

    loss = get_minibatch_loss(device, train_loader, train_step_fn)

    losses.append(loss)
