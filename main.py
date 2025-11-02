import torch
import numpy as np
import datagen2D
from datagen2D import HOGraphDataset
import DEC_HOGNN
from torch.optim.lr_scheduler import StepLR
import argparse
import DEC_HOGNN as DEC_HOGNN
import torch.nn as nn

parser = argparse.ArgumentParser(description="GPU")
parser.add_argument("--gpu", type=int, help="GPU ID", default=0)  #default:0
parser.add_argument("--mode", type=int, help="train and test:0, test only:1", default=1)  
args = parser.parse_args()

epochs, lr = 500, 1e-5
batchsize = 4
add_vol = False
grad_clip = False

def criterion_hognn(E, D, batch):
    loss = torch.einsum('ij,ij->',(E-batch.p_y[:,0:2]),(E-batch.p_y[:,0:2]))/batch.p_y.shape[0] 
    loss = loss + torch.einsum('ij,ij->',(D-batch.p_y[:,2:4]),(D-batch.p_y[:,2:4]))/batch.p_y.shape[0]
    return loss

def train(model, train_loader, optimizer, device, epoch=None):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        E,D= model(batch, integrals=False) 
        loss = criterion_hognn(E, D, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss 
    return total_loss / len(train_loader)

def eval(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            E,D= model(batch, integrals=False) 
            loss = criterion_hognn(E, D, batch)
            total_loss += loss 
    return total_loss / len(loader)
    
if __name__ == '__main__':
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    train_set = torch.load(datagen2D.path_pfx+datagen2D.file_pfx+'train_set2D.pth') 
    test_set = torch.load(datagen2D.path_pfx+datagen2D.file_pfx+'test_set2D.pth') 
    val_set = torch.load(datagen2D.path_pfx+datagen2D.file_pfx+'val_set2D.pth') 
    train_loader, test_loader, val_loader = datagen2D.get_loader(train_set, batchsize), datagen2D.get_loader(test_set, batchsize), datagen2D.get_loader(val_set, batchsize)

    model = DEC_HOGNN.DEC_HOGNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=.01)
    scheduler_step = StepLR(optimizer, step_size=1000, gamma=0.85) 
    best_val_loss = 1e9

    if args.mode == 0:
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, device, epoch=epoch)
            val_loss = eval(model, val_loader, device)
            scheduler_step.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Epoch {epoch}: train loss = {train_loss: .4f}, val loss = {val_loss: .4f}")
    
    model.load_state_dict(torch.load('best_model.pth'))
    print(f"Test loss: {eval(model, test_loader, device)}")