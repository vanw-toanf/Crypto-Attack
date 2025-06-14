import torch
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
import config
import models
from generate import generate



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
    
    NUM_PASSWORDS = 1000000
    OUTPUT_FILE = "ai_wordlist.txt"

    start_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            start_count = sum(1 for line in f)
    
    remaining_count = NUM_PASSWORDS - start_count

    if remaining_count <= 0:
        print(f"[*] File '{OUTPUT_FILE}' đã có {start_count} mật khẩu. Đã đủ.")
    else:
        print(f"[*] Đã tìm thấy {start_count} mật khẩu. Bắt đầu tạo thêm {remaining_count} mật khẩu...")
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            progress_bar = tqdm(range(remaining_count), desc="Đang tạo wordlist", initial=start_count, total=NUM_PASSWORDS)
            for _ in progress_bar:
                # Bắt đầu với một ký tự ngẫu nhiên
                start_char_idx = torch.randint(len(char_to_int) - 1, (1,)).item() # -1 để tránh chọn <EOS>
                start_char = int_to_char.get(start_char_idx, 'a')

                pwd = generate(model, char_to_int, int_to_char, start_string=start_char, temperature=0.7, top_k=10)
                
                if pwd:
                    f.write(pwd + '\n')

        print(f"\n[+] Đã tạo xong wordlist tại: {OUTPUT_FILE}")
