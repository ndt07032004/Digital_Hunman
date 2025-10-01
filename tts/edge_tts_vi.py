import time
import asyncio
import edge_tts
from tts import voice_vi
from tts.voice_vi import EnumVoice
from utils import util, config_util
from pydub import AudioSegment

class Speech:
    def __init__(self):
        # mặc định chọn giọng Hoài My
        self.voice_type = voice_vi.get_voice_of(
            config_util.config["attribute"]["voice"]
            if config_util.config["attribute"]["voice"] and config_util.config["attribute"]["voice"].strip() != ""
            else "Hoai My (edge)"
        )
        self.voice_name = EnumVoice.HOAI_MY.value["voiceName"]
        if self.voice_type is not None:
            self.voice_name = self.voice_type.value["voiceName"]

        self.__history_data = []

    def __get_history(self, voice_name, style, text):
        for data in self.__history_data:
            if data[0] == voice_name and data[1] == style and data[2] == text:
                return data[3]
        return None

    def connect(self):
        util.log(1, "Edge TTS (Vietnamese) đã kết nối miễn phí!")

    def close(self):
        pass

    async def get_edge_tts(self, text, voice, file_url) -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(file_url)

    def convert_mp3_to_wav(self, mp3_filepath):
        audio = AudioSegment.from_mp3(mp3_filepath)
        audio = audio.set_frame_rate(44100)
        wav_filepath = mp3_filepath.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_filepath, format="wav")
        return wav_filepath

    def to_sample(self, text, style):
        history = self.__get_history(self.voice_name, style, text)
        if history is not None:
            return history
        try:
            file_url = './samples/sample-' + str(int(time.time() * 1000)) + '.mp3'
            asyncio.new_event_loop().run_until_complete(self.get_edge_tts(text, self.voice_name, file_url))
            wav_url = self.convert_mp3_to_wav(file_url)
            self.__history_data.append((self.voice_name, style, text, wav_url))
            return wav_url
        except Exception as e:
            util.log(1, "[x] Chuyển đổi giọng nói thất bại!")
            util.log(1, "[x] Nguyên nhân: " + str(str(e)))
            return None

if __name__ == '__main__':
    config_util.load_config()
    sp = Speech()
    sp.connect()
    text = "Xin chào, tôi là trợ lý AI giọng nói tiếng Việt miễn phí."
    s = sp.to_sample(text, "neutral")
    print("File lưu:", s)
    sp.close()
