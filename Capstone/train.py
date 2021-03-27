# Train.py - Code for training the model

import torch
from tqdm import tqdm

def train(model, iterator, optimizer, criterion, clip, device='cuda'):
    
    model.train()
    
    epoch_loss = 0
    pbar = tqdm(iterator)
    
    
    for i, batch in enumerate(pbar):
        
        src = batch.statement
        trg = batch.code

        # print(src.shape, trg.shape)
        
        optimizer.zero_grad()

        # print(trg[:,:-1].shape)
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
