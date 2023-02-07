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

    def audio_buffer_generator():
        p = pa.PyAudio()
        stream = p.open(format=pa.paInt16,
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
        p.terminate()

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

    def limit(y1, y2):
        noise_gate = 25
        # only update indices with noticeable change
        y2 = np.where(np.abs(np.subtract(y1, y2)) > noise_gate, y2, y1)
        return savgol_filter(y2, 5, 2)

    # def animate(line1, line2):
    #     step = np.subtract(line1, line2) / (20)
    #     for i in range(20):
    #         pygame.draw.lines(window_surface, BLACK, False, line1, 1)
    #         pygame.display.update()
    #         line1 = np.add(line1, step)
    #         window_surface.fill(WHITE)

    def process_buffer():
        while run:
            try:
                frame = frames.popleft()
                xf_prev, yf_prev = get_line(frame)
                while run:
                    if frames:
                        frame = frames.popleft()
                        xf, yf = get_line(frame)
                        yf = limit(yf_prev, yf)
                        line2 = list(zip(xf.flatten(), yf.flatten()))
                        line1 = list(zip(xf_prev.flatten(), yf_prev.flatten()))
                        # threading.Thread(animate(line1, line2)).start()
                        # animate(line1, line2)
                        pygame.draw.aalines(window_surface, BLACK, False, line2, 1)
                        pygame.display.update()
                        window_surface.fill(WHITE)
                        yf_prev = yf
                        xf_prev = xf

            except IndexError:
                pass

    threading.Thread(target=audio_buffer_generator).start()
    threading.Thread(target=process_buffer).start()


if __name__ == "__main__":
    main()
