from curateData import curateData

pth = '../Commodity/Dataset/Generated/temp2.csv'

# Get the inputs and outputs from the extracted ticker data
inputs, labels, dates = curateData(pth,'Gold Futures','date',n_lags)
N = len(inputs)

# Perform the train validation split
trainX, trainY, valX, valY = train_val_split(inputs, labels, train_pct)

# Standardize the data to bring the inputs on a uniform scale
trnX, SS_ = standardizeData(trainX, train = True)
valX, _ = standardizeData(valX, SS_)

# Create dataloaders for both training and validation datasets
training_generator = getDL(trnX, trainY, params)
validation_generator = getDL(valX, valY, params)

# Create the model
model = forecasterModel(N, hidden_dim, rnn_layers, dropout).to(device)

# Define the loss function and the optimizer
loss_func = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Track the losses across epochs
train_losses = []
valid_losses = []
print("Training")

# Training loop 
for epoch in range(1, n_epochs + 1):
    ls = 0
    valid_ls = 0
    # Train for one epoch
    for xb, yb in training_generator:
        # Perform the forward pass operation
        ips = xb.unsqueeze(0)            
        targs = yb.squeeze(0)
#         targs = [int(x) for x in targs]
        op = model(ips)
#         print(targs)                
        # Backpropagate the errors through the network
        optim.zero_grad()
        loss = loss_func(op, targs)
        loss.backward()
        optim.step()
        ls += (loss.item() / ips.shape[1])
        
    # Check the performance on valiation data
    for xb, yb in validation_generator:
        ips = xb.unsqueeze(0)
        ops = model.predict(ips)
        vls = loss_func(ops, yb)
        valid_ls += (vls.item() / xb.shape[1])
        
        for i in range(len(ops)):
            if ops[i] >= 0.5:
                ops[i] = 1
            else:
                ops[i] = 0
        cm = confusion_matrix(yb.detach().numpy(), ops.detach().numpy())
        print(cm)
        
    rmse = lambda x: round(math.sqrt(x * 1.000), 3)
    train_losses.append(str(rmse(ls)))
    valid_losses.append(str(rmse(valid_ls)))
        
    # Print the total loss for every tenth epoch
    if (epoch % 10 == 0) or (epoch == 1):
        print(f"Epoch {str(epoch):<4}/{str(n_epochs):<4} | Train Loss: {train_losses[-1]:<8}| Validation Loss: {valid_losses[-1]:<8}")

# Make predictions on train, validation and test data and plot 
# the predictions along with the true values 
to_numpy = lambda x, y: (x.squeeze(0).numpy(), y.squeeze(0).numpy())
train_preds, train_labels = get_preds(training_generator, model)
train_preds, train_labels = to_numpy(train_preds, train_labels)

val_preds, val_labels = get_preds(validation_generator, model)
val_preds, val_labels = to_numpy(val_preds, val_labels)