import pyaudio as pa
import numpy as np
from collections import deque
import threading
import time
from sklearn import preprocessing as p
import scipy.fft as fft
from scipy.signal import savgol_filter
import pygame as pg


def main():
    # sets up pygame
    pg.init()

    # sets up the window
    window_surface = pg.display.set_mode((500, 500), 0, 32)
    pg.display.set_caption('Line Test')
    pg.event.get()

    # sets up the colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # draw white background
    window_surface.fill(WHITE)

    frames = deque()

    class StandardVisualiser:

        class Slider:
            def __init__(self, x, y, min_val, max_val, init_val, name):
                self.x = x
                self.y = y
                self.w = 10
                self.h = 100
                self.min = min_val
                self.max = max_val
                self.value = init_val
                self.fill = int(init_val / max_val * self.h)
                self.name = name

            def draw(self):
                pg.draw.rect(window_surface, BLACK, (self.x, self.y, self.w, self.h))
                pg.display.update()

            def update(self, value):
                self.value = value
                self.fill = int(value / self.max * self.h)
                pg.draw.rect(window_surface, BLACK, (self.x, self.y, self.w, self.fill))
                pg.display.update()

        def __init__(self):
            self.fps = 60
            self.limit = 25
            self.smooth = 7
            self.bar_width = 3
            self.run = True

        def audio_buffer_generator(self):
            buff = pa.PyAudio()
            stream = buff.open(format=pa.paInt16,
                               channels=1,
                               rate=44100,
                               input=True,
                               frames_per_buffer=1)

            while self.run:
                data = stream.read(1024, exception_on_overflow=False)
                time.sleep(1 / self.fps)
                frame_array = np.frombuffer(data, dtype=np.int16)
                frames.append(frame_array)

            stream.stop_stream()
            stream.close()
            buff.terminate()

        @staticmethod
        def get_line(local_frame):
            w, h = pg.display.get_surface().get_size()
            yf = fft.rfft(local_frame)
            xf = fft.rfftfreq(len(local_frame), 1 / 44100)
            points_per_freq = len(xf) / (44100 / 2)
            target_idx = int(points_per_freq * 1)
            yf[target_idx - 1: target_idx + 2] = 0
            xf = p.normalize(xf.reshape(1, -1)) * w * 150
            yf = p.normalize(np.abs(yf).reshape(1, -1)) * h
            yf = h - yf
            return xf, yf

        def noise_gate(self, y1, y2):
            # only update indices with noticeable change
            y2 = np.where(np.abs(np.subtract(y1, y2)) > self.limit, y2, y1)
            return savgol_filter(y2, self.smooth, 2)

        @staticmethod
        def draw_normal(line):
            pg.draw.aalines(window_surface, BLACK, False, line, 1)
            pg.display.update()
            window_surface.fill(WHITE)

        def process_buffer(self):
            while self.run:
                try:
                    frame = frames.popleft()
                    xf_prev, yf_prev = self.get_line(frame)
                    while self.run:
                        if frames:
                            frame = frames.popleft()
                            xf, yf = self.get_line(frame)
                            yf = self.noise_gate(yf_prev, yf)
                            line = list(zip(xf.flatten(), yf.flatten()))
                            # threading.Thread(self.draw_fade(line)).start()
                            self.draw_normal(line)
                            window_surface.fill(WHITE)
                            yf_prev = yf

                except IndexError:
                    pass

        def start(self):

            threading.Thread(target=self.audio_buffer_generator).start()
            threading.Thread(target=self.process_buffer).start()


    class BarVisualiser(StandardVisualiser):
        def __init__(self):
            super().__init__()
            self.bar_width = 3

        def setBarWidth(self, bar_width):
            self.bar_width = bar_width

        def draw_normal(self, line):
            w, h = pg.display.get_surface().get_size()
            for index, (x, y) in enumerate(line):
                pg.draw.rect(window_surface, BLACK, (x, y, self.bar_width, h - y))

            pg.display.update()
            window_surface.fill(WHITE)

    vs = BarVisualiser()
    vs.start()


if __name__ == "__main__":
    main()
