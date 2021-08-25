import torch
from tqdm import tqdm


def trainer(model, train_dataloader, local_rank, device, optimizer, criterion, nepochs, nepochs_save, save_path, test_dataloader=None, kinetic_lambda = 0.0):
    if local_rank == 0:
        train_accs = []
        test_accs = [] if test_dataloader is not None else None
    
    depth = len(model.module.blocks)
    v_collect = [torch.torch.empty(0) for i in range(depth)]
    def save_outputs_hook(layer_id):
        def fn(_, input, output):
            v_collect[layer_id] = output - input[0]
        return fn
    for iter_i in range(depth):
        model.module.blocks[iter_i].register_forward_hook(save_outputs_hook(iter_i))
    
    for epoch in range(nepochs):
        running_loss, running_accuracy = train(model, train_dataloader, criterion, optimizer, local_rank, device, v_collect, kinetic_lambda = kinetic_lambda)
        if local_rank == 0: 
            print(f"Epoch : {epoch+1} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")
            train_accs.append(running_accuracy)

        if test_dataloader is not None and local_rank == 0:
            test_loss, test_accuracy = evaluation(model, test_dataloader, criterion, device)
            print(f"test acc: {test_accuracy:.4f} - test loss : {test_loss:.4f}\n")
            test_accs.append(test_accuracy)
        if (epoch+1)%nepochs_save == 0 and local_rank == 0:
            # torch.save(model.state_dict(), save_path)
            torch.save({
                'epoch': epoch,
                'train_acc': train_accs,
                'test_acc': test_accs,
                'lambda': kinetic_lambda
                }, save_path) 


def train(model, dataloader, criterion, optimizer, local_rank, device, v_collect, kinetic_lambda = 0.0):
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
    
    
    if local_rank == 0: dataloader = tqdm(dataloader)
    
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        
        
        transport = kinetic_lambda * sum([torch.mean(v ** 2) for v in v_collect])
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

            output = model(data)
            loss = criterion(output, target)

            acc = (output.argmax(dim=1) == target).float().mean()
            test_accuracy += acc / len(dataloader)
            test_loss += loss.item() / len(dataloader)

    return test_loss, test_accuracy