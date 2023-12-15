import pyaudio
import wave

# 녹음 설정
FORMAT = pyaudio.paInt16  # 오디오 형식
CHANNELS = 1  # 채널 수 (1: 모노, 2: 스테레오)
RATE = 44100  # 샘플링 레이트
CHUNK = 1024  # 데이터 덩어리 크기
RECORD_SECONDS = 3  # 녹음할 시간 (초)
WAVE_OUTPUT_FILENAME = "output.wav"

audio = pyaudio.PyAudio()

# 스트림 시작
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("녹음 시작")

frames = []

# 3초간 오디오 데이터 읽기
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("녹음 완료")

# 스트림 정지 및 닫기
stream.stop_stream()
stream.close()
audio.terminate()

# 데이터를 WAV 파일로 저장
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()