from tkinter import ttk
from src.app.main_page import MainPageView

class UI:
    '''Class responsible for user interface.'''

    def __init__(self, root):
        '''
        Arguments:
            root:
                Tkinter element to create the ui in'''

        self._root = root
        self._current_view = None

        self.configure_window()

    def configure_window(self):
        self._root.resizable(width=False, height=False)

    def _hide_current_view(self):
        if self._current_view:
            self._current_view.destroy()

    def start(self):
        '''Starts ui.
        '''

        self._show_main_page()

    
    def _show_main_page(self):
        pass