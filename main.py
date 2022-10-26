




from data_extractor import *

# Global Variable
SEED = 20
GENRES_MAP = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
}
SPLIT_RATIO = 0.8
DATA_PATH = "./Data/genres_original"

# Get Data
train_x, train_y, test_x, test_y = get_data(DATA_PATH, GENRES_MAP, SPLIT_RATIO, SEED)
