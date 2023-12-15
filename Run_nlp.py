import pyaudio
import wave
import subprocess
import whisper
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

import torch
from torch import nn


w_model = whisper.load_model("base")
WAVE_OUTPUT_FILENAME = r"C:\137\output.wav"

def record_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    DEVICE_INDEX = 2  # 마이크 장치 인덱스
    

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=DEVICE_INDEX,
                        frames_per_buffer=CHUNK)

    print("녹음 시작...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("녹음 완료")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return WAVE_OUTPUT_FILENAME

def transcribe_audio(file_path):

    command = ["whisper", file_path, "--language", "ko"]
    print(command)
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

def main():
    record_audio()
    transcription = transcribe_audio(WAVE_OUTPUT_FILENAME)
    print("변환된 텍스트:")
    print(transcription)

if __name__ == "__main__":
    main()