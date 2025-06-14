import torch
import shutil
import os
import config

def save_checkpoint(state, is_best, path):
    """Lưu checkpoint và tạo bản sao cho best model."""
    if not os.path.exists(path):
        os.makedirs(path)
        
    last_filepath = os.path.join(path, 'last.pt')
    best_filepath = os.path.join(path, 'best.pt')
    
    torch.save(state, last_filepath)
    
    if is_best:
        shutil.copyfile(last_filepath, best_filepath)

def load_checkpoint(checkpoint_path, model, optimizer):
    """Tải checkpoint để tiếp tục huấn luyện."""
    if not os.path.exists(checkpoint_path):
        print(f"[!] Không tìm thấy checkpoint tại {checkpoint_path}. Bắt đầu huấn luyện từ đầu.")
        return 0, float('inf')
        
    print(f"[*] Đang tải checkpoint từ: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    
    print(f"[*] Checkpoint đã tải thành công. Tiếp tục từ epoch {start_epoch}.")
    return start_epoch, best_loss