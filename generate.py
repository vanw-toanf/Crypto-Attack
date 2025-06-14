# generate.py
import torch
import torch.nn.functional as F
import json
import os

import config
import models

def generate(model, char_to_int, int_to_char, start_string='a', max_len=16, temperature=0.8, top_k=5):
    """
    Tạo mật khẩu từ mô hình đã huấn luyện bằng Top-k sampling và ngữ cảnh đầy đủ.
    """
    model.eval()
    
    generated_password = list(start_string)
    eos_token_idx = char_to_int.get('<EOS>')
    if eos_token_idx is None:
        raise ValueError("Token <EOS> không có trong bộ từ vựng!")
    
    with torch.no_grad():
        for _ in range(max_len - len(start_string)):
            input_sequence = torch.tensor([char_to_int[s] for s in generated_password], device=config.DEVICE).unsqueeze(0)
            
            output, _ = model(input_sequence)
            last_char_logits = output[0, -1, :]
            scaled_logits = last_char_logits / temperature
            top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            sampled_index_in_top_k = torch.multinomial(top_k_probs, 1).item()
            sampled_char_index = top_k_indices[sampled_index_in_top_k].item()
            
            if sampled_char_index == eos_token_idx:
                break
            
            char = int_to_char.get(sampled_char_index)
            
            if not char: 
                break
            
            generated_password.append(char)

    return "".join(generated_password)

if __name__ == '__main__':
    map_path = os.path.join(config.MODEL_SAVE_PATH, 'char_map.json')
    with open(map_path, 'r') as f:
        maps = json.load(f)
    char_to_int = maps['char_to_int']
    
    int_to_char = {int(k): v for k, v in maps['int_to_char'].items()}

    vocab_size = len(char_to_int)

    model = models.CharRNN(vocab_size).to(config.DEVICE)
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'best.pt')
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("[*] Mô hình đã tải. Bắt đầu tạo mật khẩu...")
    print("-" * 30)
    for i in range(20): # Tạo 20 mật khẩu mẫu
        password = generate(model, char_to_int, int_to_char, start_string='admin', temperature=0.7)
        print(f"Mẫu {i+1}: {password}")
    print("-" * 30)