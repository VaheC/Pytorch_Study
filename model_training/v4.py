
epochs = 1000

losses = []

val_losses = []

for _ in range(epochs):

    loss = get_minibatch_loss(device, train_loader, train_step_fn)

    losses.append(loss)

    with torch.no_grad():
        
        val_loss = get_minibatch_loss(device, valid_loader, val_step_fn)

        val_losses.append(val_loss)
