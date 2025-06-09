
from gtts import gTTS
import pygame
import os

def speak_label(text, lang='vi'):
    print(f"🔊 Đọc: {text}")
    tts = gTTS(text=text, lang=lang)
    filename = "temp.mp3"
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(filename)
