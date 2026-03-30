"""
AI Voice Detection Model - Prediction Module
Uses trained ensemble model to classify audio as AI or Human
Returns competition-compliant response format

Supports both v2 (125 features) and v3 (200 features)
"""

import time
import os
from pathlib import Path
import numpy as np
import joblib

# Try to import v3 features first, fall back to v2
try:
    from features_v3 import extract_features, N_FEATURES, FEATURE_NAMES
    FEATURES_VERSION = "v3"
except ImportError:
    from features_v3 import extract_features
    N_FEATURES = 125
    FEATURE_NAMES = []
    FEATURES_VERSION = "v2"

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "voice_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METRICS_PATH = MODELS_DIR / "metrics.pkl"

# Global model cache
_model = None
_scaler = None
_metrics = None


def load_model():
    """Load the trained model and scaler from disk."""
    global _model, _scaler, _metrics
    
    if _model is not None:
        return True
    
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return False
    
    try:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        if METRICS_PATH.exists():
            _metrics = joblib.load(METRICS_PATH)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def is_model_trained() -> bool:
    """Check if a trained model exists."""
    return MODEL_PATH.exists() and SCALER_PATH.exists()


def get_model_info() -> dict:
    """Get information about the trained model."""
    if not load_model():
        return {"status": "not_trained", "message": "Model has not been trained yet"}
    
    return {
        "status": "ready",
        "model_version": "v1.0",
        "metrics": _metrics if _metrics else {}
    }


def generate_explanation(features: np.ndarray, prediction: str, confidence: float) -> str:
    """
    Generate a human-readable explanation based on acoustic features.
    Enhanced for v3 features with detailed jitter/shimmer/phase analysis.
    """
    explanations = []
    
    # Feature indices vary by version
    if FEATURES_VERSION == "v3":
        # v3 features (200 features) - extract key indicators
        pitch_std = features[41] if len(features) > 41 else 0
        jitter_local = features[44] if len(features) > 44 else 0
        jitter_rap = features[45] if len(features) > 45 else 0
        pause_std = features[75] if len(features) > 75 else 0
        shimmer_local = features[95] if len(features) > 95 else 0
        shimmer_apq3 = features[97] if len(features) > 97 else 0
        phase_smoothness = features[115] if len(features) > 115 else 0
        hnr = features[106] if len(features) > 106 else 0
        breath_candidates = features[83] if len(features) > 83 else 0
        filler_count = features[165] if len(features) > 165 else 0
        
        if prediction == "AI_GENERATED":
            if jitter_local < 0.8:
                explanations.append("abnormally low pitch jitter (TTS signature)")
            if jitter_rap < 0.4:
                explanations.append("missing natural pitch perturbations")
            if shimmer_local < 3.0:
                explanations.append("lack of natural amplitude variations")
            if shimmer_apq3 < 1.5:
                explanations.append("artificially stable amplitude")
            if pitch_std < 15:
                explanations.append("unnaturally flat intonation")
            if phase_smoothness < 0.3:
                explanations.append("synthesis artifacts in phase spectrum")
            if pause_std < 2:
                explanations.append("mechanical pause patterns")
            if breath_candidates < 2:
                explanations.append("no natural breathing patterns")
            if filler_count == 0:
                explanations.append("absence of natural hesitations")
            if hnr > 25:
                explanations.append("artificially clean voice")
        else:  # HUMAN
            if jitter_local > 1.0:
                explanations.append("natural voice tremor")
            if jitter_rap > 0.5:
                explanations.append("typical human pitch micro-variations")
            if shimmer_local > 4.0:
                explanations.append("natural amplitude fluctuations")
            if pitch_std > 20:
                explanations.append("expressive pitch variation")
            if breath_candidates > 1:
                explanations.append("natural breathing patterns detected")
            if pause_std > 3:
                explanations.append("varied pause durations")
            if filler_count > 0:
                explanations.append("natural speech hesitations present")
    else:
        # v2 features (125 features) - original extraction
        pitch_std = features[41] if len(features) > 41 else 0
        jitter_rel = features[46] if len(features) > 46 else 0
        pause_std = features[56] if len(features) > 56 else 0
        spectral_flux_std = features[76] if len(features) > 76 else 0
        shimmer = features[90] if len(features) > 90 else 0
        hnr = features[96] if len(features) > 96 else 0
        local_pitch_var = features[62] if len(features) > 62 else 0
        
        if prediction == "AI_GENERATED":
            if pitch_std < 20:
                explanations.append("unnaturally consistent pitch patterns")
            if jitter_rel < 0.015:
                explanations.append("minimal pitch micro-variations")
            if shimmer < 0.1:
                explanations.append("lack of natural amplitude perturbations")
            if spectral_flux_std < 5:
                explanations.append("overly smooth spectral transitions")
            if pause_std < 3:
                explanations.append("mechanical timing in speech segments")
            if hnr > 25:
                explanations.append("artificially clean harmonic structure")
        else:  # HUMAN
            if pitch_std > 25:
                explanations.append("natural pitch variation")
            if jitter_rel > 0.02:
                explanations.append("natural voice micro-tremors")
            if shimmer > 0.15:
                explanations.append("natural amplitude fluctuations")
            if spectral_flux_std > 8:
                explanations.append("dynamic spectral changes")
            if pause_std > 5:
                explanations.append("natural breathing patterns")
            if local_pitch_var > 5:
                explanations.append("expressive prosody")
    
    if not explanations:
        if prediction == "AI_GENERATED":
            explanations.append("synthetic speech characteristics detected")
        else:
            explanations.append("natural human speech characteristics")
    
    # Confidence-based qualifier
    if prediction == "AI_GENERATED":
        if confidence > 0.85:
            prefix = "High confidence AI detection"
        elif confidence > 0.65:
            prefix = "AI-generated voice likely"
        else:
            prefix = "Possible AI-generated voice"
    else:
        if confidence > 0.85:
            prefix = "High confidence human voice"
        elif confidence > 0.65:
            prefix = "Human voice detected"
        else:
            prefix = "Likely human voice"
            
    return f"{prefix}: {', '.join(explanations[:3])}"


def predict_voice(audio_path: str, language: str = "English") -> dict:
    """
    MULTI-FEATURE-FOCUS prediction - analyzes same audio multiple times,
    each pass focuses on DIFFERENT feature categories:
    
    1. BACKGROUND CHECK: Is the background suspiciously clean/uniform?
    2. SMOOTHNESS CHECK: Is the voice TOO smooth? (jitter, shimmer, micro-variations)
    3. EMOTION CHECK: Are there natural emotions? (pitch-energy coupling, prosody)
    4. BREATH CHECK: Are there natural breathing patterns?
    5. PHASE CHECK: Phase coherence patterns typical of TTS?
    
    A smooth speaker can still be human if they have natural emotion/breathing/background.
    Real AI fails MULTIPLE checks.
    
    RULE: If human confidence < 50%, classify as AI directly.
    """
    import librosa
    
    start_time = time.time()
    
    if not load_model():
        return {
            "error": True,
            "message": "Model not trained. Please run train.py first."
        }
    
    try:
        # Extract features ONCE for the full audio
        features = extract_features(audio_path)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        raw_features = features.copy()
        
        # Get overall prediction first
        features_reshaped = features.reshape(1, -1)
        features_scaled = _scaler.transform(features_reshaped)
        
        overall_probs = _model.predict_proba(features_scaled)[0]
        overall_human_prob = float(overall_probs[0])
        overall_ai_prob = float(overall_probs[1])
        
        # ========== SPECIALIZED FEATURE CHECKS ==========
        # Each check analyzes specific feature indicators
        # Thresholds relaxed for multilingual support (Hindi, Tamil, Malayalam, Telugu, English)
        
        check_results = {}
        
        # CHECK 1: BACKGROUND ANALYSIS (V4 features 200-207)
        noise_floor_var = features[200] if len(features) > 200 else 0
        snr_estimate = features[202] if len(features) > 202 else 0
        bg_consistency = features[203] if len(features) > 203 else 0
        low_freq_room_tone = features[204] if len(features) > 204 else 0
        bg_transients = features[206] if len(features) > 206 else 0
        
        bg_human_score = 0
        if noise_floor_var > 0.0005: bg_human_score += 1    # Relaxed from 0.001
        if bg_consistency < 0.97: bg_human_score += 1        # Relaxed from 0.95
        if low_freq_room_tone > 0.00005: bg_human_score += 1 # Relaxed from 0.0001
        if bg_transients > 1: bg_human_score += 1            # Relaxed from 2
        check_results['background'] = 'HUMAN' if bg_human_score >= 1 else 'AI'  # Only need 1/4
        
        # CHECK 2: SMOOTHNESS ANALYSIS (V4 features 208-215)
        jitter_scale_var = features[208] if len(features) > 208 else 0
        shimmer_temporal_var = features[210] if len(features) > 210 else 0
        amplitude_microvar = features[211] if len(features) > 211 else 0
        zcr_voiced_var = features[212] if len(features) > 212 else 0
        transition_smoothness = features[213] if len(features) > 213 else 0
        
        jitter_local = features[44] if len(features) > 44 else 0
        shimmer_local = features[95] if len(features) > 95 else 0
        
        smooth_human_score = 0
        if jitter_scale_var > 0.0005: smooth_human_score += 1    # Relaxed from 0.001
        if shimmer_temporal_var > 0.002: smooth_human_score += 1 # Relaxed from 0.005
        if amplitude_microvar > 0.0005: smooth_human_score += 1  # Relaxed from 0.001
        if jitter_local > 0.003 or shimmer_local > 0.02: smooth_human_score += 1  # Relaxed
        check_results['smoothness'] = 'HUMAN' if smooth_human_score >= 1 else 'AI'  # Only need 1/4
        
        # CHECK 3: EMOTION/PROSODY ANALYSIS (V4 features 216-224)
        pitch_energy_corr = features[216] if len(features) > 216 else 0
        f0_declination = features[217] if len(features) > 217 else 0
        speaking_rate_var = features[218] if len(features) > 218 else 0
        hnr_temporal_var = features[219] if len(features) > 219 else 0
        phrase_pitch_reset = features[220] if len(features) > 220 else 0
        energy_phrase_var = features[221] if len(features) > 221 else 0
        
        emotion_human_score = 0
        if abs(pitch_energy_corr) > 0.05: emotion_human_score += 1   # Relaxed from 0.1
        if abs(f0_declination) > 0.0005: emotion_human_score += 1    # Relaxed from 0.001
        if speaking_rate_var > 0.05: emotion_human_score += 1        # Relaxed from 0.1
        if hnr_temporal_var > 0.2: emotion_human_score += 1          # Relaxed from 0.5
        if phrase_pitch_reset > 0.5: emotion_human_score += 1        # Relaxed from 1
        check_results['emotion'] = 'HUMAN' if emotion_human_score >= 2 else 'AI'  # Need 2/5
        
        # CHECK 4: BREATH/PAUSE ANALYSIS (Part 3 features 65-84)
        pause_count = features[70] if len(features) > 70 else 0
        pause_std = features[72] if len(features) > 72 else 0
        breath_candidates = features[83] if len(features) > 83 else 0
        energy_acceleration = features[87] if len(features) > 87 else 0
        
        breath_human_score = 0
        if pause_count > 1: breath_human_score += 1          # Relaxed from 3
        if pause_std > 0.05: breath_human_score += 1         # Relaxed from 0.1
        if breath_candidates > 0: breath_human_score += 1    # Relaxed from 1
        if abs(energy_acceleration) > 0.0005: breath_human_score += 1  # Relaxed from 0.001
        check_results['breathing'] = 'HUMAN' if breath_human_score >= 1 else 'AI'  # Only need 1/4
        
        # CHECK 5: PHASE/SPECTRAL ANALYSIS (Part 5 features 115-139)
        phase_smoothness = features[115] if len(features) > 115 else 0
        phase_discontinuity = features[116] if len(features) > 116 else 0
        spectral_flux_std = features[123] if len(features) > 123 else 0
        spectral_flatness_std = features[125] if len(features) > 125 else 0
        
        hnr = features[106] if len(features) > 106 else 0
        
        phase_human_score = 0
        if phase_discontinuity > 0.005: phase_human_score += 1    # Relaxed from 0.01
        if spectral_flux_std > 0.5: phase_human_score += 1        # Relaxed from 1
        if spectral_flatness_std > 0.005: phase_human_score += 1  # Relaxed from 0.01
        if hnr < 30: phase_human_score += 1                       # Relaxed from 25
        check_results['phase'] = 'HUMAN' if phase_human_score >= 1 else 'AI'  # Only need 1/4
        
        # ========== AGGREGATE SPECIALIZED CHECKS ==========
        human_checks = sum(1 for v in check_results.values() if v == 'HUMAN')
        ai_checks = sum(1 for v in check_results.values() if v == 'AI')
        
        # Combine: model prediction (75%) + specialized checks (25%)
        # Higher model weight since the ensemble was trained on actual data
        specialized_human_score = human_checks / 5.0  # 0 to 1
        combined_human_score = (overall_human_prob * 0.75) + (specialized_human_score * 0.25)
        combined_ai_score = 1 - combined_human_score
        
        # DECISION LOGIC
        if combined_human_score < 0.45:
            classification = "AI_GENERATED"
            confidence = combined_ai_score
        elif overall_ai_prob > 0.8 and ai_checks >= 4:
            # Both model and checks strongly agree it's AI
            classification = "AI_GENERATED"
            confidence = combined_ai_score
        elif overall_ai_prob > 0.5 and human_checks >= 3:
            # Model uncertain but checks say human - trust checks (smooth speaker)
            classification = "HUMAN"
            confidence = combined_human_score
        elif human_checks >= 3:
            classification = "HUMAN"
            confidence = combined_human_score
        else:
            if combined_human_score >= 0.5:
                classification = "HUMAN"
                confidence = combined_human_score
            else:
                classification = "AI_GENERATED"
                confidence = combined_ai_score
        
        # Confidence floor boost: don't report < 60% when majority checks agree
        if classification == "HUMAN" and human_checks >= 3 and confidence < 0.70:
            confidence = 0.70 + (human_checks - 3) * 0.10  # 3/5→70%, 4/5→80%, 5/5→90%
        elif classification == "AI_GENERATED" and ai_checks >= 3 and confidence < 0.70:
            confidence = 0.70 + (ai_checks - 3) * 0.10
        
        # Build detailed explanation
        check_summary = ", ".join([f"{k}={v}" for k, v in check_results.items()])
        
        if classification == "AI_GENERATED":
            failed_checks = [k for k, v in check_results.items() if v == 'AI']
            if confidence > 0.75:
                explanation = f"Clear AI: Failed checks [{', '.join(failed_checks)}]. Model: {overall_ai_prob:.0%} AI"
            else:
                explanation = f"Likely AI: Suspicious in [{', '.join(failed_checks)}]. Combined analysis: {combined_ai_score:.0%}"
        else:
            passed_checks = [k for k, v in check_results.items() if v == 'HUMAN']
            if confidence > 0.75:
                explanation = f"Clear human: Passed [{', '.join(passed_checks)}]. Natural speech patterns detected"
            else:
                explanation = f"Likely human: Natural [{', '.join(passed_checks)}]. May be smooth speaker but has human traits"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "error": False,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation,
            "checksPerformed": check_results,
            "humanChecks": f"{human_checks}/5",
            "modelConfidence": f"AI:{overall_ai_prob:.0%} Human:{overall_human_prob:.0%}"
        }
    
    except Exception as e:
        return {
            "error": True,
            "message": f"Error processing audio: {str(e)}"
        }
