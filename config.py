# config.py

# ------------------ Optional Dependencies ------------------ #

try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except Exception:
    HAS_TORCHINFO = False


# ------------------ Model Checkpoint ------------------ #

SAVE_PATH = "/kaggle/working/Waveformer.pth"


# ------------------ System / Dataset Paths ------------------ #

class SysConfig:
    """
    Folder-based dataset structure.
    Each split (train/dev/test) contains two subfolders:
    bonafide and spoof.
    """

    TRAIN_PATH = "/kaggle/input/asv-19-aa/ASV19/train"
    DEV_PATH   = "/kaggle/input/asv-19-aa/ASV19/dev"
    TEST_PATH  = "/kaggle/input/asv-19-aa/ASV19/test"


# ------------------ Experiment Hyperparameters ------------------ #

class ExpConfig:

    # Audio Processing
    SAMPLE_RATE = 16000
    PRE_EMPHASIS = 0.97
    TRAIN_DURATION = 4   # seconds
    TEST_DURATION = 4    # seconds

    # Model
    TRANSFORMER_HIDDEN = 660

    # Training
    BATCH_SIZE = 32
    LR = 8e-4
    EPOCHS = 30


# ------------------ Config Instances ------------------ #

sys_cfg = SysConfig()
exp_cfg = ExpConfig()
