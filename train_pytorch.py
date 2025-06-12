# train_passgan_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import time
import json
import csv
from tqdm import tqdm # Thư viện để tạo thanh tiến trình (pip install tqdm)

# --- 1. CÁC THAM SỐ CẤU HÌNH ---
EPOCHS = 50
BATCH_SIZE = 64
SEQ_LENGTH = 16
LATENT_DIM = 128
EMBEDDING_DIM = 64 # Kích thước embedding cho Critic
CRITIC_LOOPS = 5
GP_WEIGHT = 10.0
MODEL_SAVE_PATH = "./passgan_models_pytorch"
# Tự động chọn thiết bị (GPU nếu có)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. HÀM TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---

class PasswordDataset(Dataset):
    """Lớp Dataset cho mật khẩu."""
    def __init__(self, data):
        self.data = torch.LongTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    """Hàm padding thủ công tương tự Keras."""
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'post':
                padded_seq = seq[:maxlen]
            else:
                padded_seq = seq[-maxlen:]
        else:
            padded_seq = seq
        
        padding_length = maxlen - len(padded_seq)
        if padding == 'post':
            padded_sequences.append(padded_seq + [value] * padding_length)
        else:
            padded_sequences.append([value] * padding_length + padded_seq)
            
    return np.array(padded_sequences)


def load_and_preprocess_data(file_path, seq_length):
    """Hàm tải dữ liệu và trả về một PyTorch DataLoader."""
    print(f"[*] Đang tải dữ liệu từ file CSV: {file_path}")
    passwords = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"    - Bỏ qua header: {header}")
            for row in reader:
                if row:
                    passwords.append(row[0])
    except FileNotFoundError:
        print(f"[!] LỖI: Không tìm thấy file {file_path}")
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

    data_as_int = [[char_to_int.get(c, 0) for c in p] for p in passwords]

    padded_data = pad_sequences(data_as_int, maxlen=seq_length, padding='post', truncating='post')
    
    dataset = PasswordDataset(padded_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    with open(os.path.join(MODEL_SAVE_PATH, 'char_map.json'), 'w') as f:
        json.dump({'char_to_int': char_to_int, 'int_to_char': int_to_char}, f)

    return dataloader, vocab_size, char_to_int, int_to_char


# --- 3. KIẾN TRÚC MÔ HÌNH ---

class Generator(nn.Module):
    def __init__(self, latent_dim, vocab_size, seq_length):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * seq_length),
            # Reshape sẽ được thực hiện trong hàm forward
            nn.LSTM(128, 128, batch_first=True, num_layers=2), # 2 lớp LSTM xếp chồng
            nn.Linear(128, vocab_size),
            nn.Softmax(dim=2)
        )

    def forward(self, z):
        x = self.model[0](z)
        x = x.view(-1, self.seq_length, 128) # Reshape
        lstm_out, _ = self.model[1](x)
        x = self.model[2](lstm_out)
        x = self.model[3](x)
        return x

class Critic(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length):
        super(Critic, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.dense = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # Lấy output của bước thời gian cuối cùng
        x = x[:, -1, :]
        x = self.dense(x)
        return x

# --- 4. HÀM TÍNH TOÁN GRADIENT PENALTY ---

def gradient_penalty(critic, real_samples, fake_samples, device):
    batch_size, seq_len = real_samples.shape
    alpha = torch.rand(batch_size, 1, 1, device=device)
    alpha = alpha.expand(-1, seq_len, -1)

    real_embeddings = critic.embedding(real_samples)
    fake_embeddings = critic.embedding(fake_samples)
    interpolated_embeddings = (alpha * real_embeddings + ((1 - alpha) * fake_embeddings)).requires_grad_(True)

    # Tạm thời tắt CuDNN để thực hiện double backward
    with torch.backends.cudnn.flags(enabled=False):
        # Chạy phần còn lại của critic trong ngữ cảnh không có CuDNN
        lstm_out1, _ = critic.lstm1(interpolated_embeddings)
        lstm_out2, _ = critic.lstm2(lstm_out1)
        interpolated_out = critic.dense(lstm_out2[:, -1, :])
    # ----------------------
    
    gradients = torch.autograd.grad(
        outputs=interpolated_out,
        inputs=interpolated_embeddings,
        grad_outputs=torch.ones_like(interpolated_out, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp


# --- 5. HÀM MAIN ĐỂ HUẤN LUYỆN ---

if __name__ == '__main__':
    print(f"[*] Sử dụng thiết bị: {DEVICE}")

    # Đường dẫn đến file CSV của bạn
    DATA_FILE_PATH = 'dataset/rockyou.csv'
    
    # Tải dữ liệu
    dataloader, vocab_size, _, _ = load_and_preprocess_data(DATA_FILE_PATH, SEQ_LENGTH)

    if dataloader is None:
        print("[!] Kết thúc chương trình do lỗi tải dữ liệu.")
    else:
        # Khởi tạo các mô hình
        generator = Generator(LATENT_DIM, vocab_size, SEQ_LENGTH).to(DEVICE)
        critic = Critic(vocab_size, EMBEDDING_DIM, SEQ_LENGTH).to(DEVICE)

        # Khởi tạo các optimizer
        optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
        optimizer_c = optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))

        print("\n[*] Bắt đầu huấn luyện...")
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            # Sử dụng tqdm để tạo thanh tiến trình
            loop = tqdm(dataloader, leave=True)
            total_c_loss = 0
            total_g_loss = 0
            
            for i, real_samples in enumerate(loop):
                real_samples = real_samples.to(DEVICE)
                batch_size = real_samples.size(0)

                # --- 1. HUẤN LUYỆN CRITIC ---
                critic.train()
                generator.eval()
                
                for _ in range(CRITIC_LOOPS):
                    optimizer_c.zero_grad()
                    
                    # Tạo nhiễu và sinh mật khẩu giả
                    z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                    with torch.no_grad():
                        fake_password_probs = generator(z)
                        fake_password_int = torch.argmax(fake_password_probs, dim=2)
                    
                    # Critic đánh giá
                    critic_real = critic(real_samples).reshape(-1)
                    critic_fake = critic(fake_password_int).reshape(-1)
                    
                    # Tính toán loss
                    gp = gradient_penalty(critic, real_samples, fake_password_int, DEVICE)
                    loss_c = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + GP_WEIGHT * gp)
                    
                    # Lan truyền ngược
                    loss_c.backward()
                    optimizer_c.step()

                # --- 2. HUẤN LUYỆN GENERATOR ---
                generator.train()
                optimizer_g.zero_grad()
                
                # Tạo nhiễu và sinh mật khẩu giả
                z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                gen_fake_probs = generator(z)
                
                # Lừa critic
                gen_fake_logits = critic(torch.argmax(gen_fake_probs, dim=2)).reshape(-1)
                loss_g = -torch.mean(gen_fake_logits)

                # Lan truyền ngược
                loss_g.backward()
                optimizer_g.step()

                # Cập nhật và hiển thị loss trên thanh tiến trình
                total_c_loss += loss_c.item()
                total_g_loss += loss_g.item()
                loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
                loop.set_postfix(
                    c_loss=loss_c.item(), 
                    g_loss=loss_g.item()
                )
            
            # In loss trung bình của epoch
            avg_c_loss = total_c_loss / len(dataloader)
            avg_g_loss = total_g_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Critic Loss: {avg_c_loss:.4f}, Avg Generator Loss: {avg_g_loss:.4f}")

        end_time = time.time()
        print(f"\n[+] Huấn luyện hoàn tất sau: {end_time - start_time:.2f} giây.")

        # Lưu lại mô hình Generator
        generator_save_path = os.path.join(MODEL_SAVE_PATH, 'passgan_generator.pth')
        torch.save(generator.state_dict(), generator_save_path)
        print(f"[+] Mô hình Generator đã được lưu tại: {generator_save_path}")