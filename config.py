# config.py
import torch

# Tham số huấn luyện
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
CRITIC_LOOPS = 2  # Số lần huấn luyện Critic cho mỗi lần huấn luyện Generator
GP_WEIGHT = 10.0  # Trọng số của Gradient Penalty

# Tham số mô hình
LATENT_DIM = 128
SEQ_LENGTH = 16
EMBEDDING_DIM = 64

# Đường dẫn
DATA_FILE_PATH = 'dataset/rockyou.csv'
MODEL_SAVE_PATH = "./passgan_models_pytorch"

# Thiết bị
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")