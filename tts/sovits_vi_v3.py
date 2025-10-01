import requests
import time
from utils import util
import wave

class Speech:
    def __init__(self):
        pass

    def connect(self):
        pass

    def close(self):
        pass

    def to_sample(self, text, style):
        url = "http://127.0.0.1:9880/tts"
        data = {
            "text": text,
            "text_lang": "vi",                  # ✅ tiếng Việt
            "ref_audio_path": "./samples/ref.wav",  
            "prompt_text": "Xin lỗi, tôi đang bận, vui lòng thử lại sau.",
            "prompt_lang": "vi",
            "top_k": 5,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": 1.2
        }
        try:
            response = requests.post(url, json=data)
            file_url = './samples/sample-' + str(int(time.time() * 1000)) + '.wav'
            if response.status_code == 200:
                with wave.open(file_url, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(32000)
                    wf.writeframes(response.content)
                return file_url
            else:
                util.log(1, "[x] Chuyển đổi giọng nói thất bại!")
                util.log(1, "[x] Nguyên nhân: " + str(response.text))
                return None
        except Exception as e:
            util.log(1, "[x] Chuyển đổi giọng nói thất bại!")
            util.log(1, "[x] Nguyên nhân: " + str(str(e)))
            return None
