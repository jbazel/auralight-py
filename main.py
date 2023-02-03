import base64

import pyaudio as pa
import numpy as np
from collections import deque
import threading
import time
from sklearn import preprocessing as p

import OpenGL
from OpenGL.raw.GL import *
from OpenGL.raw.GL.VERSION.GL_1_0 import *
from OpenGL.raw.GLU import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tkinter import *


global run
run = True

global fps
fps = 60

def main():
    frames = deque()
    ready_frames = deque()
    def audio_buffer_generator():
        p = pa.PyAudio()
        stream = p.open(format=pa.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=fps)


        while run:
            data = stream.read(1024, exception_on_overflow=False)
            time.sleep(1 / fps)
            frameArray = np.frombuffer(data, dtype=np.int16)
            frames.append(frameArray)

        stream.stop_stream()
        stream.close()
        p.terminate()


    # def audioStream():
    #     while run:
    #         time.sleep(1/fps)
    #         data = audio_buffer_generator()
    #         frameArray = np.frombuffer(next(data), dtype=np.int16)
    #         frames.append(frameArray)

    def processBuffer():
        while run:
            if frames:
                frame = np.absolute(frames.popleft())
                frame_norm = p.MinMaxScaler().fit_transform(frame.reshape(-1, 1)).flatten().tolist()
                plt.plot(frame_norm)
                plt.show()



    threading.Thread(target=audio_buffer_generator).start()
    threading.Thread(target=processBuffer).start()




if __name__ == "__main__":
    main()
