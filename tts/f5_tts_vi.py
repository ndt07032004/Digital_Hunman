from src.logger import setup_logger
from huggingface_hub import hf_hub_download
from f5_tts.infer import utils_infer  # Điều chỉnh import
import os
import time

logger = setup_logger("F5TTS")

class Speech:
    def __init__(self):
        self.model = utils_infer.load_model("DiT", dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4))
        self.vocoder = utils_infer.load_vocoder()
        self.ckpt_path = hf_hub_download("zalopay/vietnamese-tts", "model_960000.pt")
        self.vocab_file = hf_hub_download("zalopay/vietnamese-tts", "vocab.txt")

    def to_sample(self, text):
        try:
            ref_audio, ref_text = utils_infer.preprocess_ref_audio_text("./samples/ref.wav", "xin chào, tôi là hướng dẫn viên")
            final_wave, _, _ = utils_infer.infer_process(ref_audio, ref_text, text, self.model, self.vocoder, cross_fade_duration=0.15, nfe_step=32, speed=1.0)
            output_file = f"./static/audio/tts_{int(time.time() * 1000)}.wav"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'wb') as f:
                f.write(final_wave)
            return output_file
        except Exception as e:
            logger.error(f"F5TTS error: {str(e)}")
            return None

if __name__ == "__main__":
    sp = Speech()
    text = "Xin chào, tôi là TourGuideBot!"
    file_path = sp.to_sample(text)
    print(f"Audio saved at: {file_path}")