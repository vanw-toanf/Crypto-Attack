# utils.py
import torch
import shutil
import os
import config

def gradient_penalty(critic, real_samples, fake_samples):
    """
    Tính toán Gradient Penalty.
    Phép nội suy được thực hiện trong không gian embedding (không gian liên tục).
    """
    batch_size = real_samples.size(0)
    
    # 1. Lấy embeddings cho mẫu thật và giả
    real_embeddings = critic.embedding(real_samples)
    fake_embeddings = critic.embedding(fake_samples)
    
    # 2. Tạo alpha và nội suy giữa các embeddings
    # Alpha giờ sẽ có shape phù hợp để broadcast với shape của embedding
    alpha = torch.rand(batch_size, 1, 1, device=config.DEVICE)
    alpha = alpha.expand_as(real_embeddings)
    
    interpolated_embeddings = (alpha * real_embeddings + ((1 - alpha) * fake_embeddings)).requires_grad_(True)
    
    # 3. Đưa các embedding đã nội suy qua phần còn lại của Critic
    # Tạm thời tắt CuDNN để thực hiện double backward
    with torch.backends.cudnn.flags(enabled=False):
        # Chạy phần còn lại của critic trong ngữ cảnh không có CuDNN
        lstm_out, _ = critic.lstm(interpolated_embeddings)
        interpolated_out = critic.dense(lstm_out[:, -1, :])

    # 4. Tính toán gradient của output so với input là các embedding đã nội suy
    gradients = torch.autograd.grad(
        outputs=interpolated_out,
        inputs=interpolated_embeddings,
        grad_outputs=torch.ones_like(interpolated_out),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

def save_checkpoint(state, is_best, path):
    """Lưu checkpoint và tạo bản sao cho best model."""
    last_filepath = os.path.join(path, 'last.pt')
    best_filepath = os.path.join(path, 'best.pt')
    
    torch.save(state, last_filepath)
    print(f"[*] Checkpoint cuối cùng đã lưu tại: {last_filepath}")
    
    if is_best:
        shutil.copyfile(last_filepath, best_filepath)
        print(f"[*] Tìm thấy model tốt nhất, đã lưu tại: {best_filepath}")

def load_checkpoint(checkpoint_path, generator, critic, optimizer_g, optimizer_c):
    """Tải checkpoint để tiếp tục huấn luyện."""
    if not os.path.exists(checkpoint_path):
        print(f"[!] Không tìm thấy checkpoint tại {checkpoint_path}. Bắt đầu huấn luyện từ đầu.")
        return 0, float('inf')
        
    print(f"[*] Đang tải checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_c.load_state_dict(checkpoint['optimizer_c_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_g_loss = checkpoint['best_g_loss']
    
    print(f"[*] Checkpoint đã tải thành công. Tiếp tục từ epoch {start_epoch}.")
    return start_epoch, best_g_loss