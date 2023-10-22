import tkinter as tk
from tkinter import ttk, constants
from src.network.neural_network import NeuralNetwork

class MainPageView:
    '''Class for main page interface.
    '''

    def __init__(self, root):
        self._initialize()

    def pack(self):
        self._frame.pack(fill=constants.X)

    def destroy(self):
        self._frame.destroy()

    def _initialize(self):
        '''Initialize page.
        '''

        self._frame = ttk.Frame(master=self._root)
        self.heading_label = ttk.Label(master=self._frame)

    def _main(self):
        nn = NeuralNetwork()
        nn._train_network()

        test_label = ttk.Label(master=self._frame, 
                               text='ss')
        test_label.grid(row=1, column=1)
