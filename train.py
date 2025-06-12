# train.py
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import time

# Import các thành phần từ các file khác
import config
import dataset
import models
import utils

def main():
    print(f"[*] Sử dụng thiết bị: {config.DEVICE}")

    # Tải dữ liệu
    dataloader, vocab_size = dataset.load_and_preprocess_data()
    if dataloader is None:
        return

    # Khởi tạo mô hình
    generator = models.Generator(vocab_size).to(config.DEVICE)
    critic = models.Critic(vocab_size).to(config.DEVICE)

    # Khởi tạo optimizer
    print(f"[*] Cài đặt LR cho Critic: {config.LEARNING_RATE}, cho Generator: {config.LEARNING_RATE * 4}")
    optimizer_g = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE * 4, betas=(0.5, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))

    # Tải checkpoint nếu có
    start_epoch, best_g_loss = utils.load_checkpoint(
        os.path.join(config.MODEL_SAVE_PATH, 'last.pt'),
        generator, critic, optimizer_g, optimizer_c
    )

    print("\n[*] Bắt đầu huấn luyện...")
    start_time = time.time()
    
    for epoch in range(start_epoch, config.EPOCHS):
        loop = tqdm(dataloader, leave=True)
        total_c_loss = 0
        total_g_loss = 0

        for i, real_samples in enumerate(loop):
            real_samples = real_samples.to(config.DEVICE)
            batch_size = real_samples.size(0)

            # --- Huấn luyện Critic ---
            critic.train()
            generator.eval()
            for _ in range(config.CRITIC_LOOPS):
                optimizer_c.zero_grad()
                z = torch.randn(batch_size, config.LATENT_DIM, device=config.DEVICE)
                with torch.no_grad():
                    fake_password_int = torch.argmax(generator(z), dim=2)
                
                critic_real = critic(real_samples).reshape(-1)
                critic_fake = critic(fake_password_int).reshape(-1)
                gp = utils.gradient_penalty(critic, real_samples, fake_password_int)
                loss_c = -(torch.mean(critic_real) - torch.mean(critic_fake)) + config.GP_WEIGHT * gp
                loss_c.backward()
                optimizer_c.step()

            # --- Huấn luyện Generator ---
            generator.train()
            critic.train() # Đặt critic ở train() mode để có hàm backward
            optimizer_g.zero_grad()
            
            z = torch.randn(batch_size, config.LATENT_DIM, device=config.DEVICE)
            gen_fake_probs = generator(z)
            
            soft_embeddings = torch.matmul(gen_fake_probs, critic.embedding.weight)
            gen_fake_logits = critic.forward_from_embeddings(soft_embeddings).reshape(-1)
            
            loss_g = -torch.mean(gen_fake_logits)
            loss_g.backward()
            optimizer_g.step()

            total_c_loss += loss_c.item()
            total_g_loss += loss_g.item()
            loop.set_description(f"Epoch [{epoch+1}/{config.EPOCHS}]")
            loop.set_postfix(c_loss=loss_c.item(), g_loss=loss_g.item())
        
        # --- CÁC DÒNG CODE BỊ THIẾU NẰM Ở ĐÂY ---
        # Tính toán loss trung bình cho cả epoch
        avg_c_loss = total_c_loss / len(dataloader)
        avg_g_loss = total_g_loss / len(dataloader)

        # In ra thông tin tổng kết của epoch
        # Dùng \r và end='' để thanh tiến trình tqdm không bị lỗi xuống dòng
        print(f"\rEpoch [{epoch+1}/{config.EPOCHS}] - Avg Critic Loss: {avg_c_loss:.4f}, Avg Generator Loss: {avg_g_loss:.4f}")
        # ---------------------------------------------
        
        # --- Lưu Checkpoint ---
        is_best = avg_g_loss < best_g_loss
        best_g_loss = min(avg_g_loss, best_g_loss)
        
        state = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_c_state_dict': optimizer_c.state_dict(),
            'best_g_loss': best_g_loss,
        }
        utils.save_checkpoint(state, is_best, config.MODEL_SAVE_PATH)

    end_time = time.time()
    print(f"\n[+] Huấn luyện hoàn tất sau: {end_time - start_time:.2f} giây.")


if __name__ == '__main__':
    main()