import torch
from tqdm import tqdm

def trainer(model, train_dataloader, device, optimizer, criterion, nepochs, nepochs_save, save_path, test_dataloader=None, kinetic_lambda = 0.0):
    train_accs = []
    test_accs = [] if test_dataloader is not None else None
    
    for epoch in range(nepochs):
        running_loss, running_accuracy = train(model, train_dataloader, criterion, optimizer, device, kinetic_lambda = kinetic_lambda)
        print(f"Epoch : {epoch+1} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")
        train_accs.append(running_accuracy)

        if test_dataloader is not None:
            test_loss, test_accuracy = evaluation(model, test_dataloader, criterion, device)
            print(f"test acc: {test_accuracy:.4f} - test loss : {test_loss:.4f}\n")
            test_accs.append(test_accuracy)
    if (epoch+1)%nepochs_save == 0:
        torch.save({
            'epoch': epoch,
            'model': model,
            'train_acc': train_accs,
            'test_acc': test_accs
            }, save_path) 


def train(model, dataloader, criterion, optimizer, device, kinetic_lambda = 0.0):
    '''
    Function used to train the model over a single epoch and update it according to the
    calculated gradients.

    Args:
        model: Model supplied to the function
        dataloader: DataLoader supplied to the function
        criterion: Criterion used to calculate loss
        optimizer: Optimizer used update the model    

    Output:
        running_loss: Training Loss (Float)
        running_accuracy: Training Accuracy (Float)
    '''
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    for data, target in tqdm(dataloader):
        data = data.to(device)
        target = target.to(device)

        output, v_collect = model(data)
        transport = kinetic_lambda * sum([torch.mean(torch.abs(v) ** 2) for v in v_collect])
        loss = criterion(output, target) + transport
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == target).float().mean()
        running_accuracy += acc / len(dataloader)
        running_loss += loss.item() / len(dataloader)

    return running_loss, running_accuracy

def evaluation(model, dataloader, criterion, device):
    '''
    Function used to evaluate the model on the test dataset.

    Args:
        model: Model supplied to the function
        dataloader: DataLoader supplied to the function
        criterion: Criterion used to calculate loss
        
    Output:
        test_loss: Testing Loss (Float)
        test_accuracy: Testing Accuracy (Float)
    '''
    model.eval()
    with torch.no_grad():
        test_accuracy = 0.0
        test_loss = 0.0
        for data, target in tqdm(dataloader):
            data = data.to(device)
            target = target.to(device)

            output= model(data)
            loss = criterion(output, target)

            acc = (output.argmax(dim=1) == target).float().mean()
            test_accuracy += acc / len(dataloader)
            test_loss += loss.item() / len(dataloader)

    return test_loss, test_accuracy