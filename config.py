import torch

EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001

EMBEDDING_DIM = 64
HIDDEN_DIM = 256  
NUM_LAYERS = 2  


DATA_FILE_PATH = 'dataset/train.csv'
MODEL_SAVE_PATH = "./language_model_pytorch"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")