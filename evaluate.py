import torch
import torch.nn as nn
import json
import os
from tqdm import tqdm

import config
import models
from dataset import load_and_preprocess_data, PasswordLanguageModelDataset, collate_fn 
from generate import generate 

def calculate_perplexity(model, test_dataloader, vocab_size):
    """Tính toán Perplexity trên tập test."""
    print("\n[*] Bắt đầu tính toán Perplexity trên tập test...")
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    
    with torch.no_grad():
        loop = tqdm(test_dataloader, leave=True)
        for inputs, targets in loop:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
            loop.set_description("Đang tính Perplexity")

    avg_loss = total_loss / len(test_dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print("-" * 30)
    print(f"Loss trung bình trên tập test: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")
    print("-" * 30)

def calculate_uniqueness(model, char_to_int, int_to_char, num_to_generate=10000):
    """Tính toán độ duy nhất của các mật khẩu được tạo ra."""
    print(f"\n[*] Bắt đầu tạo {num_to_generate} mật khẩu để tính độ duy nhất...")
    generated_passwords = []
    
    unique_passwords = set()

    for _ in tqdm(range(num_to_generate), desc="Đang tạo mật khẩu"):
        start_char = list(char_to_int.keys())[torch.randint(len(char_to_int), (1,)).item()]
        pwd = generate(model, char_to_int, int_to_char, start_string=start_char, temperature=0.7)
        unique_passwords.add(pwd)

    uniqueness_score = len(unique_passwords) / num_to_generate
    
    print("-" * 30)
    print(f"Đã tạo: {num_to_generate} mật khẩu")
    print(f"Số lượng duy nhất: {len(unique_passwords)}")
    print(f"Tỷ lệ duy nhất: {uniqueness_score:.4%}")
    print("-" * 30)

if __name__ == '__main__':
    map_path = os.path.join(config.MODEL_SAVE_PATH, 'char_map.json')
    with open(map_path, 'r') as f:
        maps = json.load(f)
    char_to_int = maps['char_to_int']
    int_to_char = {int(k): v for k, v in maps['int_to_char'].items()}
    vocab_size = len(char_to_int)

    model = models.CharRNN(vocab_size).to(config.DEVICE)
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pt')
    state_dict = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state_dict)


    # Tính Perplexity
    original_data_path = config.DATA_FILE_PATH
    config.DATA_FILE_PATH = 'dataset/test.csv'
    test_dataloader, _, _, _ = load_and_preprocess_data()
    config.DATA_FILE_PATH = original_data_path 
    
    if test_dataloader:
        calculate_perplexity(model, test_dataloader, vocab_size)

    # Tính Uniqueness
    calculate_uniqueness(model, char_to_int, int_to_char)
    