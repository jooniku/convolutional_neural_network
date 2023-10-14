from tkinter import Tk
from app.ui import UI


def main():
    '''Main window for application.
    This function is called with poetry command.
    '''

    window = Tk()
    window.title('Convolutional Neural Network App')

    ui_view = UI(window)
    ui_view.start()

    window.mainloop()


if __name__ == '__main__':
    main()
