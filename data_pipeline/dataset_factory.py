# data_pipeline/dataset_factory.py
# Auto-generated placeholder
import os
import random
import json
import torchaudio
import torch

class AsianALMDatasetGenerator:
    def __init__(self, output_dir="generated_dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulating source libraries (In real implementation, these point to actual folders)
        self.speech_sources = {
            "Hindi": ["demo_hindi.wav"],
            "Mandarin": ["demo_mandarin.wav"],
            "Tamil": ["demo_tamil.wav"]
        }
        self.noise_sources = {
            "Airport": ["announcement.wav", "jet_engine.wav"],
            "Traffic": ["honk.wav", "engine_idling.wav"]
        }

    def mix_audio(self, speech_path, noise_path, snr_db=10):
        """
        Superimposes speech onto non-speech events to create the 'Joint' signal.
        """
        # (Simplified mixing logic for prototype)
        # Real implementation would use torchaudio.functional.add_noise with SNR calculation
        return "mixed_audio_path.wav" 

    def generate_reasoning_annotation(self, language, noise_type):
        """
        Generates the 'Thinking' logic (QnA) that OpenAQA lacks for this specific context.
        """
        # In a real pipeline, an LLM (GPT-4) would generate these dynamically based on metadata.
        
        templates = {
            ("Hindi", "Airport"): {
                "Q": "What can be inferred from the audio?",
                "A": "The background features jet engine sounds and announcements, indicating an airport. The speech is in Hindi, suggesting the speaker is likely an Indian traveler at the boarding gate."
            },
            ("Mandarin", "Traffic"): {
                "Q": "Describe the environment and the speaker.",
                "A": "Loud honking and engine noises suggest a busy highway. The speaker is speaking Mandarin, likely stuck in traffic in a Chinese-speaking region."
            }
        }
        return templates.get((language, noise_type), {"Q": "Analyze audio", "A": "General audio scene."})

    def create_sample(self, index):
        """Creates one training example: Audio + JSON Label"""
        lang = random.choice(list(self.speech_sources.keys()))
        noise = random.choice(list(self.noise_sources.keys()))
        
        # 1. Mix Audio
        mixed_file = f"sample_{index}.wav"
        # self.mix_audio(...) # Actual mixing would happen here
        
        # 2. Generate Ground Truth (The reasoning)
        annotation = self.generate_reasoning_annotation(lang, noise)
        
        entry = {
            "id": index,
            "audio_path": mixed_file,
            "language": lang,
            "event_context": noise,
            "question": annotation["Q"],
            "answer": annotation["A"] # This is the target for the LLM
        }
        
        return entry

    def build_dataset(self, num_samples=100):
        print(f"--- 🏭 Generating 'Asian-ALM-Instruct' Dataset ({num_samples} samples) ---")
        dataset_json = []
        for i in range(num_samples):
            entry = self.create_sample(i)
            dataset_json.append(entry)
        
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(dataset_json, f, indent=2)
        print(f"--- ✅ Dataset generated at /{self.output_dir} ---")

if __name__ == "__main__":
    generator = AsianALMDatasetGenerator()
    generator.build_dataset(10) 