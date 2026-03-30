# training_modules/config.py

class TrainingConfig:
    # --- PATHS ---
    USER_DATA_PATH = "training/knowledge_base.json"
    SYNTHETIC_DATA_PATH = "training/combined_dataset.json"
    OUTPUT_DIR = "backend/trained_aura_model"
    
    # --- MODEL SETTINGS ---
    BASE_MODEL = "google/flan-t5-small"
    
    # --- HYPERPARAMETERS ---
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 4
    EPOCHS = 3           # How many times to loop over the data
    TEST_SIZE = 0.1      # 10% of data used for validation