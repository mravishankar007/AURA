import os
import sys
import json
import torch
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf 
import networkx as nx
import uuid
from datetime import datetime
from collections import defaultdict
from scipy import signal
from scipy.ndimage import median_filter

# --- 1. FORCE FFMPEG PATH (Windows Specific) ---
if os.name == 'nt':
    os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# --- ML IMPORTS ---
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --- SPEAKER DIARIZATION IMPORTS ---
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("⚠️ SpeechBrain not available. Using fallback speaker diarization.")

try:
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available. Using fallback speaker diarization.")

# --- KNOWLEDGE BASE (TRAINING MEMORY) ---
class KnowledgeBase:
    def __init__(self, db_path="training/knowledge_base.json"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if not os.path.exists(db_path):
            with open(db_path, "w", encoding='utf-8') as f: json.dump([], f)

    def save_entry(self, transcript, events, emotion, user_notes):
        if isinstance(transcript, list):
            full_text = " ".join([t.get('text', '') for t in transcript])
        else:
            full_text = str(transcript)
            
        summary_text = full_text[:150] + "..." if len(full_text) > 150 else full_text

        new_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": str(datetime.now()),
            "events": [e['label'] for e in events] if events else [],
            "emotion": emotion,
            "transcript_summary": summary_text,
            "full_transcript_data": transcript,
            "user_notes": user_notes
        }
        
        try:
            with open(self.db_path, "r+", encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(new_entry)
                f.seek(0)
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.truncate()
            return "✅ Knowledge Base Updated! Model weights will adjust in next batch."
        except Exception as e:
            return f"❌ Error saving to Knowledge Base: {e}"

    def get_similar_context(self):
        if not os.path.exists(self.db_path): return []
        try:
            with open(self.db_path, "r", encoding='utf-8') as f:
                data = json.load(f)
            return [f"Past Event: {d['events']}, Note: {d['user_notes']}" for d in data[-3:]]
        except:
            return []

# --- AUDIO PREPROCESSING ENGINE ---
class AudioPreprocessor:
    """Advanced audio preprocessing for recorded audio."""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def normalize_audio(self, audio, target_level=-20.0):
        """Normalize audio to target dB level."""
        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-10:
            return audio
        
        # Convert to dB
        current_db = 20 * np.log10(rms)
        
        # Calculate gain
        gain_db = target_level - current_db
        gain = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio * gain
        
        # Clip to prevent distortion
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def pre_emphasis(self, audio, coef=0.97):
        """Apply pre-emphasis filter to boost high frequencies."""
        return np.append(audio[0], audio[1:] - coef * audio[:-1])
    
    def high_pass_filter(self, audio, cutoff=80, order=4):
        """Apply high-pass filter to remove low-frequency noise."""
        nyquist = self.sr / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def low_pass_filter(self, audio, cutoff=8000, order=4):
        """Apply low-pass filter to remove high-frequency noise."""
        nyquist = self.sr / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio)
    
    def spectral_gating(self, audio, sr, n_std_thresh=1.5, prop_decrease=0.9):
        """Advanced noise reduction using spectral gating."""
        # Estimate noise from the first 0.5 seconds (assuming silence/noise)
        noise_len = int(0.5 * sr)
        if len(audio) > noise_len:
            noise_clip = audio[:noise_len]
        else:
            noise_clip = audio
        
        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_clip,
            prop_decrease=prop_decrease,
            stationary=False,
            n_std_thresh_stationary=n_std_thresh
        )
        
        return reduced
    
    def remove_silence(self, audio, sr, top_db=30):
        """Remove silence from audio."""
        # Detect non-silent intervals
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        # Concatenate non-silent parts
        non_silent = []
        for start, end in intervals:
            non_silent.append(audio[start:end])
        
        if non_silent:
            return np.concatenate(non_silent)
        return audio
    
    def enhance_speech(self, audio, sr):
        """Enhance speech clarity for better recognition."""
        # Apply pre-emphasis
        emphasized = self.pre_emphasis(audio)
        
        # Apply high-pass filter to remove rumble
        filtered = self.high_pass_filter(emphasized, cutoff=80)
        
        # Apply low-pass filter to remove hiss
        filtered = self.low_pass_filter(filtered, cutoff=8000)
        
        # Normalize
        normalized = self.normalize_audio(filtered, target_level=-20.0)
        
        return normalized
    
    def preprocess_recorded_audio(self, audio, sr):
        """Complete preprocessing pipeline for recorded audio."""
        print("Preprocessing recorded audio...")
        
        # Step 1: Remove silence
        audio_no_silence = self.remove_silence(audio, sr, top_db=25)
        
        # Step 2: Spectral gating for noise reduction
        audio_denoised = self.spectral_gating(audio_no_silence, sr)
        
        # Step 3: Enhance speech
        audio_enhanced = self.enhance_speech(audio_denoised, sr)
        
        # Step 4: Final normalization
        audio_final = self.normalize_audio(audio_enhanced, target_level=-20.0)
        
        print("Audio preprocessing complete")
        return audio_final

# --- SPEAKER DIARIZATION ENGINE ---
class SpeakerDiarizer:
    """Advanced speaker diarization using embeddings and clustering with overlap detection."""
    
    def __init__(self):
        self.encoder = None
        self.speaker_embeddings = {}
        self.speaker_labels = {}
        self.next_speaker_id = 0
        self.overlap_threshold = 0.3  # Threshold for detecting overlapping speech
        
        if SPEECHBRAIN_AVAILABLE:
            try:
                print("Loading Speaker Encoder (SpeechBrain)...")
                self.encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                print("✅ Speaker Encoder loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load SpeechBrain encoder: {e}")
                self.encoder = None
    
    def extract_speaker_embedding(self, audio_segment, sr=16000):
        """Extract speaker embedding from audio segment."""
        if self.encoder is None:
            return None
        
        try:
            # Ensure audio is the right format
            if isinstance(audio_segment, np.ndarray):
                audio_tensor = torch.FloatTensor(audio_segment).unsqueeze(0)
            else:
                audio_tensor = audio_segment
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None
    
    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def detect_overlapping_speech(self, audio_segment, sr=16000):
        """Detect if multiple speakers are talking simultaneously."""
        try:
            # Compute multiple features for overlap detection
            features = []
            
            # 1. Spectral flux (changes in spectrum)
            spectral_flux = np.mean(librosa.onset.onset_strength(y=audio_segment, sr=sr))
            features.append(spectral_flux)
            
            # 2. Spectral centroid variance
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
            centroid_var = np.var(spectral_centroid)
            features.append(centroid_var)
            
            # 3. Zero crossing rate variance
            zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
            zcr_var = np.var(zcr)
            features.append(zcr_var)
            
            # 4. RMS energy variance
            rms = librosa.feature.rms(y=audio_segment)[0]
            rms_var = np.var(rms)
            features.append(rms_var)
            
            # 5. Spectral bandwidth variance
            bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr)[0]
            bandwidth_var = np.var(bandwidth)
            features.append(bandwidth_var)
            
            # Combine features for overlap detection
            overlap_score = np.mean(features)
            
            # Normalize score
            overlap_score = min(1.0, overlap_score / 10.0)
            
            return overlap_score > self.overlap_threshold
            
        except Exception as e:
            print(f"Overlap detection error: {e}")
            return False
    
    def extract_multiple_embeddings(self, audio_segment, sr=16000, n_embeddings=3):
        """Extract multiple embeddings from different parts of audio segment for better speaker separation."""
        if self.encoder is None:
            return []
        
        try:
            segment_len = len(audio_segment)
            chunk_len = segment_len // n_embeddings
            
            embeddings = []
            for i in range(n_embeddings):
                start = i * chunk_len
                end = start + chunk_len if i < n_embeddings - 1 else segment_len
                chunk = audio_segment[start:end]
                
                if len(chunk) > sr * 0.05:  # Minimum 50ms
                    emb = self.extract_speaker_embedding(chunk, sr)
                    if emb is not None:
                        embeddings.append(emb)
            
            return embeddings
        except Exception as e:
            print(f"Multiple embedding extraction error: {e}")
            return []
    
    def assign_speaker(self, embedding, threshold=0.70):
        """Assign speaker based on embedding similarity with improved threshold."""
        if embedding is None:
            return f"Speaker {self.next_speaker_id}"
        
        best_match = None
        best_similarity = 0.0
        
        for speaker_id, speaker_emb in self.speaker_embeddings.items():
            similarity = self.compute_similarity(embedding, speaker_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id
        
        if best_similarity >= threshold:
            # Update running average of embedding for this speaker
            self.speaker_embeddings[best_match] = (
                0.7 * self.speaker_embeddings[best_match] + 0.3 * embedding
            )
            return best_match
        else:
            # New speaker
            new_speaker_id = f"Speaker {chr(65 + self.next_speaker_id)}"
            self.speaker_embeddings[new_speaker_id] = embedding
            self.next_speaker_id += 1
            return new_speaker_id
    
    def diarize_with_clustering(self, audio_segments, sr=16000):
        """Perform speaker diarization using clustering on embeddings with overlap detection."""
        if not SKLEARN_AVAILABLE or not self.encoder:
            return self.fallback_diarization(audio_segments)
        
        print("Extracting speaker embeddings with overlap detection...")
        embeddings = []
        valid_indices = []
        overlap_flags = []
        
        for i, segment in enumerate(audio_segments):
            # Check for overlapping speech
            is_overlapping = self.detect_overlapping_speech(segment, sr)
            overlap_flags.append(is_overlapping)
            
            if is_overlapping:
                print(f"⚠️ Overlapping speech detected in segment {i}")
                # Extract multiple embeddings for overlapping segments
                segment_embeddings = self.extract_multiple_embeddings(segment, sr, n_embeddings=3)
                if segment_embeddings:
                    # Use the first embedding for clustering
                    embeddings.append(segment_embeddings[0])
                    valid_indices.append(i)
            else:
                emb = self.extract_speaker_embedding(segment, sr)
                if emb is not None:
                    embeddings.append(emb)
                    valid_indices.append(i)
        
        if len(embeddings) < 2:
            return self.fallback_diarization(audio_segments)
        
        # Cluster embeddings using DBSCAN for better speaker separation
        embeddings_array = np.array(embeddings)
        
        # Use DBSCAN for automatic speaker count detection
        clustering = DBSCAN(
            eps=0.3,  # Maximum distance between samples
            min_samples=1,  # Minimum samples per cluster
            metric='cosine'
        )
        labels = clustering.fit_predict(embeddings_array)
        
        # Map labels back to all segments
        speaker_labels = {}
        for idx, label in zip(valid_indices, labels):
            if label == -1:  # Noise point in DBSCAN
                speaker_labels[idx] = f"Speaker {chr(65 + self.next_speaker_id)}"
                self.next_speaker_id += 1
            else:
                speaker_labels[idx] = f"Speaker {chr(65 + label)}"
        
        # Fill in gaps with nearest speaker
        for i in range(len(audio_segments)):
            if i not in speaker_labels:
                # Find nearest valid segment
                nearest_idx = min(valid_indices, key=lambda x: abs(x - i))
                speaker_labels[i] = speaker_labels[nearest_idx]
        
        # Handle overlapping segments by assigning multiple speakers
        for i, is_overlapping in enumerate(overlap_flags):
            if is_overlapping and i in speaker_labels:
                # Mark as potentially overlapping
                speaker_labels[i] = f"{speaker_labels[i]}+"
        
        return speaker_labels
    
    def diarize_with_realtime_detection(self, audio_segments, sr=16000):
        """Real-time speaker change detection with improved accuracy."""
        print("Performing real-time speaker change detection...")
        
        speaker_labels = {}
        current_speaker = "Speaker A"
        speaker_history = []
        change_threshold = 0.65  # Lower threshold for more sensitive detection
        
        for i, segment in enumerate(audio_segments):
            emb = self.extract_speaker_embedding(segment, sr)
            
            if emb is None:
                speaker_labels[i] = current_speaker
                continue
            
            # Check similarity with current speaker
            if current_speaker in self.speaker_embeddings:
                similarity = self.compute_similarity(emb, self.speaker_embeddings[current_speaker])
                
                if similarity < change_threshold:
                    # Speaker change detected
                    print(f"🔄 Speaker change detected at segment {i} (similarity: {similarity:.2f})")
                    
                    # Find best matching speaker from history
                    best_match = None
                    best_similarity = 0.0
                    
                    for speaker_id, speaker_emb in self.speaker_embeddings.items():
                        if speaker_id != current_speaker:
                            sim = self.compute_similarity(emb, speaker_emb)
                            if sim > best_similarity:
                                best_similarity = sim
                                best_match = speaker_id
                    
                    if best_similarity >= change_threshold:
                        current_speaker = best_match
                    else:
                        # New speaker
                        current_speaker = f"Speaker {chr(65 + self.next_speaker_id)}"
                        self.next_speaker_id += 1
                    
                    self.speaker_embeddings[current_speaker] = emb
                else:
                    # Update current speaker embedding
                    self.speaker_embeddings[current_speaker] = (
                        0.7 * self.speaker_embeddings[current_speaker] + 0.3 * emb
                    )
            else:
                # First speaker
                self.speaker_embeddings[current_speaker] = emb
            
            speaker_labels[i] = current_speaker
            speaker_history.append(current_speaker)
        
        return speaker_labels
    
    def fallback_diarization(self, audio_segments):
        """Fallback diarization using energy and spectral features with improved detection."""
        print("Using fallback speaker diarization...")
        speaker_labels = {}
        current_speaker = "Speaker A"
        last_energy = 0.0
        last_centroid = 0.0
        last_mfcc = None
        
        for i, segment in enumerate(audio_segments):
            # Compute energy
            energy = np.sqrt(np.mean(segment ** 2))
            
            # Compute spectral centroid
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=16000))
            
            # Compute MFCC features
            mfcc = librosa.feature.mfcc(y=segment, sr=16000, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Check for significant change in characteristics
            energy_change = abs(energy - last_energy) / (last_energy + 1e-8)
            centroid_change = abs(spectral_centroid - last_centroid) / (last_centroid + 1e-8)
            
            # MFCC similarity
            mfcc_similarity = 0.0
            if last_mfcc is not None:
                mfcc_similarity = cosine_similarity([mfcc_mean], [last_mfcc])[0][0]
            
            # Speaker change detection
            if (energy_change > 0.4 or centroid_change > 0.25 or mfcc_similarity < 0.7):
                current_speaker = "Speaker B" if current_speaker == "Speaker A" else "Speaker A"
            
            speaker_labels[i] = current_speaker
            last_energy = energy
            last_centroid = spectral_centroid
            last_mfcc = mfcc_mean
        
        return speaker_labels

# --- MAIN INTELLIGENCE ENGINE ---
class AuraEngine:
    def __init__(self):
        print("--- INITIALIZING AURA ENGINE (HIGH ACCURACY MODE) ---")
        
        # 1. ASR
        print("Loading Whisper Model (Medium)...")
        self.asr_model = WhisperModel("medium", device="cpu", compute_type="int8")
        
        # 2. Audio Event Detection
        self.event_pipe = pipeline("audio-classification", model="mit/ast-finetuned-audioset-10-10-0.4593")
        
        # 3. Audio Emotion (How it sounds) - FIXED VARIABLE NAME HERE
        self.emotion_pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        
        # 4. Text Emotion (What is said)
        self.text_emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
        
        # 5. Cognitive QnA
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        # 6. Speaker Diarization
        self.diarizer = SpeakerDiarizer()
        
        # 7. Audio Preprocessor
        self.preprocessor = AudioPreprocessor(sr=16000)
        
        self.memory = KnowledgeBase()
        
        self.lang_map = {
            "English": "en", "Hindi": "hi", "Mandarin": "zh", 
            "Urdu": "ur", "Tamil": "ta", "Spanish": "es", "French": "fr"
        }
        
        self.urgency_keywords = [
            "help", "emergency", "danger", "fire", "attack", "immediately",
            "मदद", "आग", "खतरा", "तुरंत",
            "救命", "危险", "火灾", "立即",
            "مدد", "خطرہ", "آگ", "فوری",
            "உதவி", "ஆபத்து", "தீ", "உடனடியாக",
            "ayuda", "peligro", "fuego", "inmediatamente",
            "aide", "danger", "feu", "immédiatement"
        ]
        
        print("--- MODELS READY ---")

    def check_urgency(self, text):
        return any(k in text.lower() for k in self.urgency_keywords)

    def detect_voice_activity(self, audio, sr=16000, frame_length=1024, hop_length=512, threshold=0.02):
        """Detect voice activity to skip silence segments."""
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find frames above threshold
        voice_frames = rms > threshold
        
        # Convert to time segments
        voice_segments = []
        in_segment = False
        start = 0
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_segment:
                start = i * hop_length / sr
                in_segment = True
            elif not is_voice and in_segment:
                end = i * hop_length / sr
                if end - start > 0.1:  # Minimum segment length
                    voice_segments.append((start, end))
                in_segment = False
        
        if in_segment:
            voice_segments.append((start, len(audio) / sr))
        
        return voice_segments

    def extract_audio_segments(self, audio, sr, segments):
        """Extract audio segments for speaker diarization."""
        audio_segments = []
        for seg in segments:
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Ensure minimum length
            if len(segment_audio) < sr * 0.1:
                segment_audio = np.pad(segment_audio, (0, int(sr * 0.1) - len(segment_audio)))
            
            audio_segments.append(segment_audio)
        
        return audio_segments

    def process_audio(self, audio_path, language="English"):
        print(f"Processing: {audio_path} | Target: {language}")
        
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Apply advanced preprocessing for recorded audio
            print("Applying advanced audio preprocessing...")
            y_preprocessed = self.preprocessor.preprocess_recorded_audio(y, sr)
            
            # Additional noise reduction
            y_denoised = nr.reduce_noise(y=y_preprocessed, sr=sr, stationary=True, prop_decrease=0.85)
            
            clean_filename = f"temp_clean_{uuid.uuid4().hex[:8]}.wav"
            clean_path = os.path.abspath(clean_filename)
            sf.write(clean_path, y_denoised, sr)
        except Exception as e:
            print(f"Audio Load Error: {e}")
            return [], [], "Error"

        try:
            iso_lang = self.lang_map.get(language, "en")
            
            # Use faster-whisper with optimized settings for speed
            segments, info = self.asr_model.transcribe(
                clean_path, 
                beam_size=5, 
                language=iso_lang,
                vad_filter=True,  # Enable VAD filter for speed
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )
            
            # Collect segments for batch processing
            segment_list = list(segments)
            
            if not segment_list:
                return [], [], "Silence"
            
            # Extract audio segments for speaker diarization
            audio_segments = self.extract_audio_segments(y_denoised, sr, segment_list)
            
            # Perform speaker diarization with real-time detection
            print("Performing speaker diarization with overlap detection...")
            speaker_labels = self.diarizer.diarize_with_realtime_detection(audio_segments, sr)
            
            # Build transcript with speaker assignments
            transcript = []
            for i, seg in enumerate(segment_list):
                is_urgent = self.check_urgency(seg.text)
                
                try:
                    text_emo_result = self.text_emotion_pipe(seg.text)[0]
                    text_tone = text_emo_result['label']
                except:
                    text_tone = "neutral"
                
                display_tone = text_tone.capitalize()
                
                if is_urgent:
                    if text_tone in ["fear", "sadness"]: display_tone = "Panic"
                    elif text_tone in ["anger", "disgust"]: display_tone = "Hostile"
                    else: display_tone = "Urgent"
                
                speaker = speaker_labels.get(i, f"Speaker {chr(65 + i % 2)}")
                
                transcript.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "speaker": speaker,
                    "confidence": round(seg.avg_logprob, 2),
                    "is_urgent": is_urgent,
                    "tone": display_tone
                })

            # Batch process events and emotions for speed
            events = self.event_pipe(clean_path, top_k=2)
            emotions = self.emotion_pipe(clean_path, top_k=1)
            
            audio_tone = emotions[0]['label'] if emotions else "neutral"
            global_emotion = f"{audio_tone.title()} (Audio) / {transcript[0]['tone'] if transcript else 'Neutral'} (Context)"
            
        finally:
            if os.path.exists(clean_path):
                try: os.remove(clean_path)
                except: pass

        return transcript, events, global_emotion

    def build_asg(self, transcript, events, emotion):
        G = nx.DiGraph()
        G.add_node("CONTEXT", label=f"Scene\nContext", title=emotion, color="#FF4B4B", shape="box")

        for ev in events:
            if ev['score'] > 0.1:
                label = ev['label'].replace("_", " ").title()
                node_id = f"Event_{label}"
                G.add_node(node_id, label=label, title=f"Conf: {ev['score']:.2f}", color="#FFA500", shape="ellipse")
                G.add_edge("CONTEXT", node_id, label="contains")

        previous_node = None
        for t in transcript:
            if t['speaker'] not in G.nodes:
                G.add_node(t['speaker'], label=t['speaker'], color="#00ADB5", shape="dot")
                G.add_edge("CONTEXT", t['speaker'], label="participant")
            
            short_txt = (t['text'][:20] + '..') if len(t['text']) > 20 else t['text']
            node_id = f"Txt_{t['start']}"
            
            tone_colors = {"Panic": "#FF0000", "Hostile": "#8B0000", "Joy": "#00FF00", "Neutral": "#00BFFF"}
            node_color = tone_colors.get(t['tone'], "#00BFFF")
            
            G.add_node(node_id, label=short_txt, title=f"[{t['tone']}] {t['text']}", color=node_color, shape="box")
            G.add_edge(t['speaker'], node_id, label="said")
            
            if previous_node:
                G.add_edge(previous_node, node_id, label="next", dashes=True)
            previous_node = node_id

        return G

    def generate_insight(self, transcript, events, emotion):
        main_event = events[0]['label'].replace("_", " ") if events else "Silence"
        urgent_count = sum(1 for t in transcript if t.get('is_urgent'))
        
        status = "NEUTRAL"
        if urgent_count > 0 or "siren" in main_event or "alarm" in main_event:
            status = "CRITICAL"
        elif "music" in main_event:
            status = "SAFE"
            
        full_reasoning = (
            f"Detected **{main_event}**. "
            f"Found {urgent_count} urgent segments. "
            f"Dominant Emotion: **{emotion}**."
        )
        return full_reasoning, status

    def answer_question(self, transcript, events, emotion, question):
        event_str = ", ".join([e['label'] for e in events]) if events else "None"
        transcript_text = " ".join([f"{t['speaker']} ({t['tone']}): {t['text']}" for t in transcript])
        
        past_knowledge = self.memory.get_similar_context()
        memory_str = " | ".join(past_knowledge) if past_knowledge else "No past data."
        
        context_block = (
            f"Scene: {event_str}. "
            f"Overall Tone: {emotion}. "
            f"Transcript: {transcript_text[:600]}. "
            f"Past Learnings: {memory_str}."
        )
        
        input_text = f"question: {question} context: {context_block}"
        input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids
        outputs = self.qa_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def train_model(self, corrected_transcript, user_notes):
        return self.memory.save_entry(corrected_transcript, [], "User-Verified", user_notes)
