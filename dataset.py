# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import json
import os
import config

class PasswordDataset(Dataset):
    """Lớp Dataset cho mật khẩu."""
    def __init__(self, data):
        self.data = torch.LongTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def pad_sequences(sequences, maxlen, value=0):
    """Hàm padding thủ công."""
    padded_sequences = [p + [value] * (maxlen - len(p)) for p in sequences]
    return np.array(padded_sequences)

def load_and_preprocess_data():
    """Hàm tải dữ liệu và trả về một PyTorch DataLoader."""
    print(f"[*] Đang tải dữ liệu từ file CSV: {config.DATA_FILE_PATH}")
    passwords = []
    try:
        with open(config.DATA_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"    - Bỏ qua header: {header}")
            for row in reader:
                if row:
                    passwords.append(row[0])
    except FileNotFoundError:
        print(f"[!] LỖI: Không tìm thấy file {config.DATA_FILE_PATH}")
        return None, None, None, None

    chars = sorted(list(set("".join(passwords))))
    if not chars:
        print("[!] LỖI: Không tìm thấy mật khẩu nào trong file.")
        return None, None, None, None

    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    print(f"    - Tìm thấy {len(passwords)} mật khẩu.")
    print(f"    - Kích thước từ vựng: {vocab_size} ký tự.")

    data_as_int = [[char_to_int.get(c, 0) for c in p if len(p) <= config.SEQ_LENGTH] for p in passwords]

    padded_data = pad_sequences(data_as_int, maxlen=config.SEQ_LENGTH)
    
    dataset = PasswordDataset(padded_data)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)
    with open(os.path.join(config.MODEL_SAVE_PATH, 'char_map.json'), 'w') as f:
        json.dump({'char_to_int': char_to_int, 'int_to_char': int_to_char}, f)

    return dataloader, vocab_size