# training_modules/trainer_reasoning.py
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from .config import TrainingConfig
from .data_manager import DataManager

class ReasoningTrainer:
    def __init__(self):
        self.cfg = TrainingConfig()
        self.data_manager = DataManager()
        
    def train(self):
        print("\n=== 🚀 STARTING MODULAR TRAINING (REASONING ENGINE) ===")
        
        # 1. Prepare Data
        data_path = self.data_manager.prepare_training_set()
        with open(data_path, "r") as f:
            raw_data = json.load(f)
            
        hf_dataset = Dataset.from_list(raw_data)
        dataset_split = hf_dataset.train_test_split(test_size=self.cfg.TEST_SIZE)
        
        # 2. Load Model & Tokenizer
        print(f"--- 🧠 Loading Base Model: {self.cfg.BASE_MODEL} ---")
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.BASE_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.BASE_MODEL)
        
        # 3. Tokenize Data
        def preprocess(examples):
            inputs = [str(x) for x in examples["input_text"]]
            targets = [str(x) for x in examples["target_text"]]
            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
            labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_data = dataset_split.map(preprocess, batched=True)
        
        # 4. Configure Training
        args = Seq2SeqTrainingArguments(
            output_dir="./results_temp",
            evaluation_strategy="epoch",
            learning_rate=self.cfg.LEARNING_RATE,
            per_device_train_batch_size=self.cfg.BATCH_SIZE,
            num_train_epochs=self.cfg.EPOCHS,
            save_total_limit=1,
            logging_steps=5,
        )
        
        # 5. Run Training
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            tokenizer=tokenizer,
        )
        
        print("--- 🏋️ Training in Progress... ---")
        trainer.train()
        
        # 6. Export Module
        print(f"--- 💾 Saving Module to {self.cfg.OUTPUT_DIR} ---")
        model.save_pretrained(self.cfg.OUTPUT_DIR)
        tokenizer.save_pretrained(self.cfg.OUTPUT_DIR)
        print("=== ✅ TRAINING COMPLETE ===")

if __name__ == "__main__":
    trainer = ReasoningTrainer()
    trainer.train()