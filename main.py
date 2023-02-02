import pyaudio
import numpy as np

def main():
    def audio_buffer_generator():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)
            
        while True:
            data = stream.read(1024)
            yield data
            stream.stop_stream()
            stream.close()
            p.terminate()

    audio_buffer = audio_buffer_generator()
    print(audio_buffer)
    

if __name__ == "__main__":
    main()