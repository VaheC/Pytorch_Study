
lr = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

model = nn.Sequential(nn.Linear(1, 1)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)

lossfn = nn.MSELoss(reduction="mean")

train_step_fn = make_train_step_fn(model, lossfn, optimizer)
