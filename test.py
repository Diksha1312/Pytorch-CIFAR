
#from utils import progress_bar
import torch

def test(epoch, num_epochs, model, testloader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            train_loss = loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f'Test Loss: {loss:.3f} | Test Acc: {100.*correct/total:.3f}')
            

            
    acc = 100.0 * correct/total
    return acc