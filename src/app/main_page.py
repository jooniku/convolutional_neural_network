import tkinter as tk
from tkinter import ttk, constants
from src.network.neural_network import NeuralNetwork
import numpy as np
from PIL import Image


class MainPageView:
    '''Class for main page interface.
    '''

    def __init__(self, root):
        self._initialize()

    def pack(self):
        self._frame.pack(fill=constants.X)

    def destroy(self):
        self._frame.destroy()

    def initialize(self):
        '''Initialize page.
        '''

        self._frame = ttk.Frame(master=self._root)
        self.heading_label = ttk.Label(master=self._frame,
                                       text="Number identifier")

    def main(self):
        nn = NeuralNetwork()
        nn.train_network()

        test_label = ttk.Label(master=self._frame,
                               text='ss')
        test_label.grid(row=1, column=1)

    def create_canvas(self):
        canvas = tk.Canvas(master=self._frame, width=200,
                           height=200, bg="white")
        canvas.pack()
