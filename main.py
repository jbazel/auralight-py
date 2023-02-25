import pyaudio as pa
import numpy as np
from collections import deque
import threading
import time
import os
from sklearn import preprocessing as p
import scipy.fft as fft
from scipy.signal import savgol_filter, butter, filtfilt
import pygame as pg

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.properties import NumericProperty


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

        def __init__(self):
            self.fps = 60
            self.limit = 150
            self.smooth = 7
            self.bar_width = 3
            self.scale = 150
            self.run = True

            self.fps_slider = Slider(min=1, max=60, value=60, step=1, size_hint=(0.5, 0.1))
            self.fps_slider.bind(value=self.set_fps)

            self.limit_slider = Slider(min=1, max=150, value=150, step=1, size_hint=(0.5, 0.1))
            self.limit_slider.bind(value=self.set_limit)

            self.smooth_slider = Slider(min=2, max=20, value=7, step=1, size_hint=(0.5, 0.1))
            self.smooth_slider.bind(value=self.set_smooth)

            self.scale_slider = Slider(min=50, max=250, value=50, step=10, size_hint=(0.5, 0.1))
            self.scale_slider.bind(value=self.set_scale)

            self.app = App()
            self.app.root = GridLayout(cols=2, rows=4)
            self.app.root.add_widget(Label(text="FPS"))
            self.app.root.add_widget(self.fps_slider)
            self.app.root.add_widget(Label(text="Limit"))
            self.app.root.add_widget(self.limit_slider)
            self.app.root.add_widget(Label(text="Smooth"))
            self.app.root.add_widget(self.smooth_slider)
            self.app.root.add_widget(Label(text="Scale"))
            self.app.root.add_widget(self.scale_slider)

        def set_fps(self, instance, value):
            self.fps = value

        def set_limit(self, instance, value):
            self.limit = value

        def set_smooth(self, instance, value):
            self.smooth = value

        def set_scale(self, instance, value):
            self.scale = value

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

        def get_line(self, local_frame):
            w, h = pg.display.get_surface().get_size()
            yf = fft.rfft(local_frame)
            xf = fft.rfftfreq(len(local_frame), 1 / 44100)
            points_per_freq = len(xf) / (44100 / 2)
            target_idx = int(points_per_freq * 1)
            yf[target_idx - 1: target_idx + 2] = 0
            b, a = butter(8, 0.8, 'high', analog=False)
            yf = filtfilt(b, a, yf)
            xf = p.normalize(xf.reshape(1, -1)) * w * self.scale
            yf = p.normalize(np.abs(yf).reshape(1, -1)) * h * 0.8
            yf = h - yf

            return xf, yf

        def noise_gate(self, y1, y2):
            noise_floor = 10
            # only update indices with noticeable change

            y2 = np.where(np.abs(np.subtract(y1, y2)) > self.limit, y2, y1)
            y2 = np.where(y2 > noise_floor, y2, 0)

            # high pass filter
            return savgol_filter(y2, self.smooth, 2)

        # static method to draw the line
        @staticmethod
        def draw_normal(line):
            pg.draw.aalines(window_surface, BLACK, False, line, 1)
            pg.display.update()
            window_surface.fill(WHITE)

        # function to process the audio buffer
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
            self.app.run()

    # class definition for the bar visualiser
    # inherits from the standard visualiser

    class BarVisualiser(StandardVisualiser):
        def __init__(self):
            super().__init__()
            self.bar_width = 3

        def setBarWidth(self, bar_width):
            self.bar_width = bar_width

        def draw_normal(self, line):
            w, h = pg.display.get_surface().get_size()
            for index, (x, y) in enumerate(line):
                pg.draw.rect(window_surface, BLACK, (x*self.scale/w, y, self.bar_width, h - y))

            pg.display.update()
            window_surface.fill(WHITE)

    # class definition for the circle visualiser
    # inherits from the standard visualiser

    class CircleVisualiser(StandardVisualiser):
        def __init__(self):
            super().__init__()
            self.circle_radius = 3

        def setCircleRadius(self, circle_radius):
            self.circle_radius = circle_radius

        def draw_normal(self, line):
            for index, (x, y) in enumerate(line):
                pg.draw.circle(window_surface, BLACK, (x, y), self.circle_radius)

            pg.display.update()
            window_surface.fill(WHITE)

    vs = CircleVisualiser()
    vs.start()


if __name__ == "__main__":
    main()
