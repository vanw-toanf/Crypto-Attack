import torch
from torch.utils.data import Dataset, DataLoader
import csv
import json
import os
import config

class PasswordLanguageModelDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Input là tất cả trừ ký tự cuối
        # Target là tất cả trừ ký tự đầu
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1]), torch.tensor(sequence[1:])

def load_and_preprocess_data():
    print(f"[*] Đang tải dữ liệu từ file CSV: {config.DATA_FILE_PATH}")
    passwords = []
    try:
        with open(config.DATA_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row and 1 < len(row[0]) < 50: # loc mat khau 5-50 ky tu
                    passwords.append(row[0])
    except FileNotFoundError:
        return None, None, None

    chars = sorted(list(set("".join(passwords))))
    vocab = chars + ['<EOS>']
    
    char_to_int = {ch: i for i, ch in enumerate(vocab)}
    int_to_char = {i: ch for i, ch in enumerate(vocab)}
    vocab_size = len(vocab)
    eos_token_idx = char_to_int['<EOS>']
    
    sequences = [[char_to_int[c] for c in p] + [eos_token_idx] for p in passwords]
   

    print(f"    - Tìm thấy {len(passwords)} mật khẩu.")
    print(f"    - Kích thước từ vựng: {vocab_size} ký tự.")


    dataset = PasswordLanguageModelDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)
    with open(os.path.join(config.MODEL_SAVE_PATH, 'char_map.json'), 'w') as f:
        json.dump({'char_to_int': char_to_int, 'int_to_char': int_to_char}, f)

    return dataloader, vocab_size, char_to_int, int_to_char

def collate_fn(batch):
    """Hàm xử lý padding cho mỗi batch."""
    inputs, targets = zip(*batch)
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_inputs, padded_targets