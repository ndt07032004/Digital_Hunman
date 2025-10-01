from src.logger import setup_logger
from huggingface_hub import hf_hub_download
import os
import time
from src.config import TTS_CONFIG
import edge_tts
import asyncio

logger = setup_logger("TTS")

class TTSIntegrator:
    def __init__(self):
        self.use_huggingface = True
        try:
            # Thử repo VinAI PhoVITS
            self.ckpt_path = hf_hub_download("vinai/PhoVITS-Vietnamese-TTS", "model.pt")
            self.vocab_file = hf_hub_download("vinai/PhoVITS-Vietnamese-TTS", "vocab.txt")
            logger.info("Loaded VinAI PhoVITS Vietnamese TTS model from Hugging Face.")
        except Exception as e:
            logger.error(f"Failed to download VinAI model: {str(e)}. Falling back to Edge TTS.")
            self.use_huggingface = False
            self.voice = "vi-VN-HoaiMyNeural"

    async def text_to_speech(self, text):
        if self.use_huggingface:
            try:
                from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process, preprocess_ref_audio_text
                model = load_model("DiT", dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4), ckpt_path=self.ckpt_path)
                vocoder = load_vocoder()
                ref_audio, ref_text = preprocess_ref_audio_text("./samples/ref.wav", "xin chào")
                final_wave, _, _ = infer_process(ref_audio, ref_text, text, model, vocoder, cross_fade_duration=0.15, nfe_step=32, speed=1.0)
                output_file = os.path.join(TTS_CONFIG['output_dir'], f"tts_{int(time.time() * 1000)}.wav")
                os.makedirs(TTS_CONFIG['output_dir'], exist_ok=True)
                with open(output_file, 'wb') as f:
                    f.write(final_wave)
                return output_file
            except Exception as e:
                logger.error(f"Hugging Face TTS error: {str(e)}. Falling back to Edge TTS.")
                self.use_huggingface = False

        # Fallback to Edge TTS
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            output_file = os.path.join(TTS_CONFIG['output_dir'], f"tts_{int(time.time() * 1000)}.mp3")
            os.makedirs(TTS_CONFIG['output_dir'], exist_ok=True)
            await communicate.save(output_file)
            return output_file
        except Exception as e:
            logger.error(f"Edge TTS error: {str(e)}")
            return None

def run_tts(text):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(TTSIntegrator().text_to_speech(text))
    finally:
        loop.close()