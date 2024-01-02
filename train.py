#from utils import progress_bar

def train(epoch, num_epochs, model, trainloader, device, optimizer, criterion):
    print(f"\nEpoch {epoch}\{num_epochs}:")
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f'Train Loss: {loss:.3f} | Train Acc: {100.*correct/total:.3f}')

