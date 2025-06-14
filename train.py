import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import time

import config
import dataset
import models
from torch.utils.tensorboard import SummaryWriter
import utils


def main():
    dataloader, vocab_size, char_to_int, _ = dataset.load_and_preprocess_data()
    if dataloader is None: return
    padding_idx = 0 

    model = models.CharRNN(vocab_size).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx) 
    
    writer = SummaryWriter(f'runs/password_model_{int(time.time())}')
    
    start_epoch, best_loss = utils.load_checkpoint(
        os.path.join(config.MODEL_SAVE_PATH, 'last.pt'),
        model, 
        optimizer
    )

    best_loss = float('inf')
    print("\nStart ....")
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, leave=True)

        for inputs, targets in loop:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            outputs, _ = model(inputs)
            
            # Reshape output và target để phù hợp với CrossEntropyLoss
            # Output: (batch * seq_len, vocab_size)
            # Target: (batch * seq_len)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{config.EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"\rEpoch [{epoch+1}/{config.EPOCHS}] - Avg Loss: {avg_loss:.4f}")
        
        writer.add_scalar('Training Loss', avg_loss, epoch)

        is_best = avg_loss < best_loss
        best_loss = min(avg_loss, best_loss)
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        utils.save_checkpoint(state, is_best, config.MODEL_SAVE_PATH)

    end_time = time.time()
    writer.close()
    print(f"\n[+] Complete: {end_time - start_time:.2f}s.")

if __name__ == '__main__':
    main()