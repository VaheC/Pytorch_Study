
epochs = 1000

losses = []

val_losses = []

for epoch in range(epochs):

    loss = get_minibatch_loss(device, train_loader, train_step_fn)

    losses.append(loss)

    with torch.no_grad():
        
        val_loss = get_minibatch_loss(device, valid_loader, val_step_fn)

        val_losses.append(val_loss)

    writer.add_scalars(
        main_tag='loss',
        tag_scalar_dict={'training': loss, 'validation': val_loss},
        global_step=epoch
    )

writer.close()
