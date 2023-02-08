import base64
import pyaudio as pa
import numpy as np
from collections import deque
import threading
import time
from sklearn import preprocessing as p
import scipy.fft as fft
from scipy.signal import savgol_filter
import pygame

global run
run = True

global fps
fps = 60


def main():
    # sets up pygame
    pygame.init()

    # sets up the window
    window_surface = pygame.display.set_mode((500, 500), 0, 32)
    pygame.display.set_caption('Line Test')

    # sets up the colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # draw white background
    window_surface.fill(WHITE)

    frames = deque()

    class StandardVisualiser:
        @staticmethod
        def audio_buffer_generator():
            buff = pa.PyAudio()
            stream = buff.open(format=pa.paInt16,
                               channels=1,
                               rate=44100,
                               input=True,
                               frames_per_buffer=1)

            while run:
                data = stream.read(1024, exception_on_overflow=False)
                time.sleep(1 / fps)
                frame_array = np.frombuffer(data, dtype=np.int16)
                frames.append(frame_array)

            stream.stop_stream()
            stream.close()
            buff.terminate()

        @staticmethod
        def get_line(local_frame):
            w, h = pygame.display.get_surface().get_size()
            yf = fft.rfft(local_frame)
            xf = fft.rfftfreq(len(local_frame), 1 / 44100)
            points_per_freq = len(xf) / (44100 / 2)
            target_idx = int(points_per_freq * 1)
            yf[target_idx - 1: target_idx + 2] = 0
            xf = p.normalize(xf.reshape(1, -1)) * w * 200
            yf = p.normalize(np.abs(yf).reshape(1, -1)) * h
            yf = h - yf
            return xf, yf

        @staticmethod
        def limit(y1, y2):
            noise_gate = 25
            # only update indices with noticeable change
            y2 = np.where(np.abs(np.subtract(y1, y2)) > noise_gate, y2, y1)
            return savgol_filter(y2, 5, 2)

        def process_buffer(self):
            while run:
                try:
                    frame = frames.popleft()
                    xf_prev, yf_prev = self.get_line(frame)
                    while run:
                        if frames:
                            frame = frames.popleft()
                            xf, yf = self.get_line(frame)
                            yf = self.limit(yf_prev, yf)
                            line = list(zip(xf.flatten(), yf.flatten()))
                            pygame.draw.aalines(window_surface, BLACK, False, line, 1)
                            pygame.display.update()
                            window_surface.fill(WHITE)
                            yf_prev = yf

                except IndexError:
                    pass

        def start(self):
            threading.Thread(target=self.audio_buffer_generator).start()
            threading.Thread(target=self.process_buffer).start()

    vs = StandardVisualiser()
    vs.start()


if __name__ == "__main__":
    main()
