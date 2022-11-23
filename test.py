from tkinter import *
from tkinter.filedialog import askopenfilename

windows = Tk()
windows.title("File Dialog Example")
windows.geometry("500x500")


def file_open():
    text_window.delete('1.0', END)
    filePath = askopenfilename(
        initialdir='C:/', title='Select a File', filetype=(("Text File", ".txt"), ("All Files", "*.*")))
    with open(filePath, 'r+') as askedFile:
        fileContents = askedFile.read()

    text_window.insert(INSERT, fileContents)
    print(filePath)


open_button = Button(windows, text="Open File", command=file_open).grid(row=4, column=3)
text_window = Text(windows, bg="white",width=200, height=150)
text_window.place(x=50, y=50)

windows.mainloop()