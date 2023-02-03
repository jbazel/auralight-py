import pyaudio as pa
import numpy as np
from collections import deque
import threading
import time
def main():
    q = deque()
    def audio_buffer_generator():
        p = pa.PyAudio()
        stream = p.open(format=pa.paInt16,
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


    def audioStream():

        while True:
            time.sleep(1/60)
            q.append(audio_buffer_generator())

    def processBuffer():
        while True:
            if q:
                data = q.popleft()
                print(data.get_audio_features())


    threading.Thread(target=audioStream).start()
    threading.Thread(target=processBuffer).start()



if __name__ == "__main__":
    main()
