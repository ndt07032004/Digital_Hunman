from enum import Enum

class EnumVoice(Enum):
    # Edge Vietnamese voices
    HOAI_MY = {
        "name": "Hoai My (edge)",
        "voiceName": "vi-VN-HoaiMyNeural",
        "styleList": {
            "neutral": "general",
            "cheerful": "cheerful",
            "calm": "calm"
        }
    }
    NAM_MINH = {
        "name": "Nam Minh (edge)",
        "voiceName": "vi-VN-NamMinhNeural",
        "styleList": {
            "neutral": "general",
            "cheerful": "cheerful",
            "calm": "calm"
        }
    }

def get_voice_list():
    return [
        EnumVoice.HOAI_MY,
        EnumVoice.NAM_MINH
    ]

def get_voice_of(name):
    for voice in get_voice_list():
        v = voice.value
        if v["name"] == name:
            return voice
    return None
