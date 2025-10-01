import requests
import time
from src.logger import setup_logger
import wave
import os

logger = setup_logger("SoVITSv3")

class Speech:
    def to_sample(self, text):
        url = "http://127.0.0.1:9880/tts"
        data = {
            "text": text, "text_lang": "vi", "ref_audio_path": "./samples/ref.wav",
            "prompt_text": "Xin chào, tôi là hướng dẫn viên", "prompt_lang": "vi",
            "top_k": 5, "top_p": 1, "temperature": 1, "text_split_method": "cut5",
            "batch_size": 1, "batch_threshold": 0.75, "split_bucket": True,
            "speed_factor": 1.0, "fragment_interval": 0.3, "seed": -1,
            "media_type": "wav", "streaming_mode": False, "parallel_infer": True,
            "repetition_penalty": 1.2
        }
        try:
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                output_file = f"./static/audio/sovits_v3_{int(time.time() * 1000)}.wav"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with wave.open(output_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(32000)
                    wf.writeframes(response.content)
                return output_file
            else:
                logger.error(f"SoVITSv3 failed: {response.text}")
                return None
        except Exception as e:
            logger.error(f"SoVITSv3 error: {str(e)}")
            return None

if __name__ == "__main__":
    sp = Speech()
    text = "Chào mừng đến với Việt Nam!"
    file_path = sp.to_sample(text)
    print(f"Audio saved at: {file_path}")