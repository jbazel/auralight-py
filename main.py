import base64

import menu as menu
import pyaudio as pa
import numpy as np
from collections import deque
import threading
import time
from sklearn import preprocessing as p
import scipy.fft as fft
from scipy.signal import savgol_filter
import pygame
import pygame_menu as pgm

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


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
            w, h = pygame.display.get_surface().get_size()
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
            pygame.draw.aalines(window_surface, BLACK, False, line, 1)
            pygame.display.update()
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

        def set_FPS(self):
            self.fps = self.menu.get_size()

        def set_limit(self):
            self.limit = self.menu.get_input_data()

        def set_smooth(self):
            self.smooth = self.menu.get_input_data()

        def start(self):
            fig, ax = plt.subplots()
            ax_fps = fig.add_axes([0.25, 0.1, 0.65, 0.03])
            fps_slider = Slider(
                ax=ax_fps,
                label='Frequency [Hz]',
                valmin=0.1,
                valmax=30,
                valinit=60,
            )

            fps_slider.on_changed(self.set_FPS)


            threading.Thread(target=self.audio_buffer_generator).start()
            threading.Thread(target=self.process_buffer).start()

    class BarVisualiser(StandardVisualiser):
        def __init__(self):
            super().__init__()
            self.bar_width = 3

        def setBarWidth(self, bar_width):
            self.bar_width = bar_width

        def draw_normal(self, line):
            w, h = pygame.display.get_surface().get_size()
            for index, (x, y) in enumerate(line):
                pygame.draw.rect(window_surface, BLACK, (x, y, self.bar_width, h - y))

            pygame.display.update()
            window_surface.fill(WHITE)

    vs = BarVisualiser()
    vs.start()


if __name__ == "__main__":
    main()
