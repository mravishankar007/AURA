# training_modules/data_manager.py
import json
import os
import random
from .config import TrainingConfig

class DataManager:
    def __init__(self):
        self.cfg = TrainingConfig()
        
    def load_user_data(self):
        """Loads real data captured from the UI"""
        if not os.path.exists(self.cfg.USER_DATA_PATH):
            return []
        try:
            with open(self.cfg.USER_DATA_PATH, "r") as f:
                return json.load(f)
        except:
            return []

    def generate_synthetic_data(self, num_samples=50):
        """Generates base knowledge so the model isn't empty"""
        scenarios = [
            {"events": ["siren", "traffic"], "tone": "urgent", "q": "Safety status?", "a": "Critical. Emergency detected."},
            {"events": ["birds", "wind"], "tone": "calm", "q": "Where is this?", "a": "Outdoors, likely nature setting."},
            {"events": ["keyboard", "mouse"], "tone": "neutral", "q": "Activity?", "a": "Office work or typing."}
        ]
        data = []
        for _ in range(num_samples):
            s = random.choice(scenarios)
            # Format: "question: [Q] context: [Context]" -> "[Answer]"
            events_str = ", ".join(s['events'])
            context = f"Sounds: {events_str}. Tone: {s['tone']}."
            data.append({
                "input_text": f"question: {s['q']} context: {context}",
                "target_text": s['a']
            })
        return data

    def prepare_training_set(self):
        """Merges User Data + Synthetic Data"""
        print("--- 🔄 DATA MODULE: Merging Datasets ---")
        
        # 1. Get Synthetic
        dataset = self.generate_synthetic_data()
        
        # 2. Get User Data (Active Learning)
        user_entries = self.load_user_data()
        print(f"   - Found {len(user_entries)} user examples.")
        
        for entry in user_entries:
            # Convert UI format to Training format
            events = ", ".join(entry['events'])
            context = f"Sounds: {events}. Tone: {entry['emotion']}."
            # We assume the user wants the model to analyze the scene
            dataset.append({
                "input_text": f"question: Analyze this scene. context: {context}",
                "target_text": entry['notes'] # The human correction
            })
            
        # 3. Save Final Dataset
        os.makedirs(os.path.dirname(self.cfg.SYNTHETIC_DATA_PATH), exist_ok=True)
        with open(self.cfg.SYNTHETIC_DATA_PATH, "w") as f:
            json.dump(dataset, f, indent=4)
            
        print(f"--- ✅ DATA READY: {len(dataset)} samples saved. ---")
        return self.cfg.SYNTHETIC_DATA_PATH