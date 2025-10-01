import requests
import time
from src.logger import setup_logger
import wave
import os

logger = setup_logger("SoVITS")

class Speech:
    def to_sample(self, text):
        url = "http://127.0.0.1:9880"
        data = {"text": text, "text_language": "vi", "cut_punc": ",."}
        try:
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                output_file = f"./static/audio/sovits_{int(time.time() * 1000)}.wav"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with wave.open(output_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(response.content)
                return output_file
            else:
                logger.error(f"SoVITS failed: {response.text}")
                return None
        except Exception as e:
            logger.error(f"SoVITS error: {str(e)}")
            return None

if __name__ == "__main__":
    sp = Speech()
    text = "Chào bạn, tôi là TourGuideBot!"
    file_path = sp.to_sample(text)
    print(f"Audio saved at: {file_path}")