import wave

import pyaudio as pa
import numpy as np
from collections import deque
import threading
import time
import os

import pygame
from sklearn import preprocessing as p
import scipy.fft as fft
from scipy.signal import savgol_filter, butter, filtfilt
import pygame as pg

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.properties import NumericProperty



class AudioProcessor:
    def __init__(self, filename, visualizer):
        self.visualizer = visualizer
        self.filename = filename
        self.data = wave.open(self.filename, 'rb')
        self.audio = pa.PyAudio()
        self.stream = self.audio.open(format=self.audio.get_format_from_width(self.data.getsampwidth()), channels=self.data.getnchannels(), rate=self.data.getframerate(), output=True)

    def play(self):
        start = time.time()
        data = self.data.readframes(1024)
        while data:
            clock.tick()


            frame = np.frombuffer(data, dtype=np.int16)
            xf, yf = self.fft(frame)
            self.visualizer.draw(xf, yf)

            self.stream.write(data)
            data = self.data.readframes(1024)


    def fft(self, frame):
        yf = fft.rfft(frame)
        xf = fft.rfftfreq(len(frame), 1 / 44100)

        points_per_freq = len(xf) / (44100 / 2)
        target_idx = int(points_per_freq * 1)
        yf[target_idx - 1: target_idx + 2] = 0

        return xf, yf
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()



class VisualizationEngine:
    def __int__(self):
        self.fps = 60
        self.limit = 150
        self.smooth = 7
        self.bar_width = 3
        self.scale = 150
        self.run = True
        self.log_scale = []

    def get_fps(self):
        fps = str(int(clock.get_fps()))
        fps_text = font.render(fps, 1, pygame.Color("coral"))
        return fps_text

    def set_fps(self, fps):
        self.fps = fps

    def draw(self, xf, yf):

        w, h = pg.display.get_surface().get_size()
        xf = xf[:int(len(xf)/2)]
        yf = yf[:int(len(yf)/2)]
        count = 40
        maximum = len(xf)
        step = (np.emath.log(maximum))/(count -1)

        filter_indices = [int(np.e**(x*step)) -1 for x in range(count)]
        filter_indices = list(dict.fromkeys(filter_indices))
        count = len(filter_indices)
        
        frequencies = xf[filter_indices]

        amplitudes = yf[filter_indices]


        x_points = [int((w/count) * i) for i in range(count)]


        amplitudes = p.normalize(np.abs(amplitudes).reshape(1, -1)) * h * 0.75 + (h / 4)
        amplitudes = h - amplitudes

        # line = list(zip(x_points, amplitudes.flatten()))
        line = list(zip(x_points, amplitudes.flatten()))

        for index, (x, y) in enumerate(line):
            pg.draw.circle(window_surface, (0, 0, 0), (x, y), 10)

        window_surface.blit(self.get_fps(), (50, 50))
        pg.display.update()
        window_surface.fill(WHITE)



if __name__ == "__main__":
    pg.init()
    font = pg.font.SysFont("Arial", 18)
    clock = pg.time.Clock()
    window_surface = pg.display.set_mode((1500, 500), pygame.RESIZABLE, 32)
    pg.display.set_caption('Line Test')
    pg.event.get()

    # sets up the colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # draw white background
    window_surface.fill(WHITE)


    ve = VisualizationEngine()
    test = AudioProcessor("test.wav", ve)
    test.play()
