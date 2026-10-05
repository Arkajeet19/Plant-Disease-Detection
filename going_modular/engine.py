import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model:torch.nn.Module,
                loss_fn:torch.nn.Module,
                data_loader:torch.utils.data.DataLoader,
                Optimizer:torch.optim.Optimizer,
                device):

    model.train()
    train_loss, train_acc =0.0, 0.0
    
    for batch,(X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        
        y_logit = model(X)
        
        loss = loss_fn(y_logit, y)
        
        train_loss += loss.item()
        
        Optimizer.zero_grad()
        
        loss.backward()
        
        Optimizer.step()
        
        y_pred = torch.argmax(torch.softmax(y_logit, dim=1),dim=1)
        
        train_acc += ((y_pred==y).sum().item()/len(y_pred))
        
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    return train_loss, train_acc

def test_step(model, loss_fn, data_loader, device:torch.device):
    test_loss, test_acc = 0.0, 0.0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_logit = model(X)
            loss = loss_fn(test_logit, y)
            test_loss += loss.item()
            test_pred = torch.argmax(torch.softmax(test_logit,dim=1),dim=1)
            test_acc += ((test_pred==y).sum().item()/len(test_pred))
    
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    return test_loss,test_acc


def train(model: torch.nn.Module,
          train_data_loader:torch.utils.data.DataLoader,
          test_data_loader:torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          epochs:int,
          device:torch.device,
          scheduler=None):
    
    results ={
        "train_loss":[],
        "train_acc":[],
        "test_loss":[],
        "test_acc":[]
    }
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc =train_step(model=model,
                                            loss_fn=loss_fn,
                                            data_loader=train_data_loader,
                                            Optimizer=optimizer,
                                            device=device)
        test_loss, test_acc =test_step(model=model,
                                        loss_fn=loss_fn,
                                        data_loader=test_data_loader,
                                        device=device)

        if scheduler is not None:
            scheduler.step()
        
        print(
            f"Epoch:{epoch+1} |"
            f"Train Loss:{train_loss} |"
            f"Train Accuracy:{train_acc} |"
            f"Test Loss:{test_loss} |"
            f"Test Accuracy:{test_acc} |"
            f"LR:{optimizer.param_groups[0]['lr']}"
        )
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
