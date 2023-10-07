import tkinter as tk
from tkinter import ttk, constants


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


