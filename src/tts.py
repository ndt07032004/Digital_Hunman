from src.logger import setup_logger
from huggingface_hub import hf_hub_download
import f5_tts.infer.utils_infer as f5_utils  # Sử dụng alias để import trực tiếp
from src.config import TTS_CONFIG
import os
import time

logger = setup_logger("TTS")

class TTSIntegrator:
    def __init__(self):
        try:
            self.ckpt_path = hf_hub_download("zalopay/vietnamese-tts", "model_960000.pt")  # Thử tải checkpoint
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {str(e)}. Using default or local path.")
            self.ckpt_path = "path/to/local/model_960000.pt"  # Thay bằng đường dẫn cục bộ nếu có
            if not os.path.exists(self.ckpt_path):
                raise ValueError("Checkpoint not found. Please download manually or check repository.")
        
        self.vocab_file = hf_hub_download("zalopay/vietnamese-tts", "vocab.txt")
        self.model = f5_utils.load_model("DiT", dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4), ckpt_path=self.ckpt_path)
        self.vocoder = f5_utils.load_vocoder()

    def text_to_speech(self, text):
        try:
            ref_audio, ref_text = f5_utils.preprocess_ref_audio_text("./samples/ref.wav", "cả hai bên hãy cố gắng hiểu cho nhau")
            final_wave, _, _ = f5_utils.infer_process(ref_audio, ref_text, text, self.model, self.vocoder, cross_fade_duration=0.15, nfe_step=32, speed=1.0)
            output_file = os.path.join(TTS_CONFIG['output_dir'], f"tts_{int(time.time() * 1000)}.wav")
            os.makedirs(TTS_CONFIG['output_dir'], exist_ok=True)
            with open(output_file, 'wb') as f:
                f.write(final_wave)
            return output_file
        except Exception as e:
            logger.error(f"TTS processing error: {str(e)}")
            return None

def run_tts(text):
    return TTSIntegrator().text_to_speech(text)