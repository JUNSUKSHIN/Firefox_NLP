import pyaudio
import wave

def list_audio_devices(p):
    """오디오 장치 목록을 출력합니다."""
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

def record_audio(device_id):
    """특정 오디오 장치에서 오디오를 녹음합니다."""
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    # 사용자가 선택한 오디오 장치로 스트림 시작
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=device_id,
                        frames_per_buffer=CHUNK)

    print("녹음 시작")

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

# 오디오 장치 목록을 출력하고 사용자 선택을 받습니다.
audio = pyaudio.PyAudio()
list_audio_devices(audio)

device_id = int(input("사용할 오디오 장치 ID를 입력하세요: "))

# 사용자가 선택한 오디오 장치에서 녹음을 시작합니다.
record_audio(device_id)
