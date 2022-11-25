import tkinter
import tkinter.filedialog

def OnDirectoryButtonClick(self):
    self.entryVariable_dir.set( tkinter.filedialog.askdirectory() )
    self.entry_dir.focus_set()
    self.entry_dir.selection_range(0, tkinter.END)

def OnFileButtonClick(self):
    self.entryVariable_file.set( tkinter.filedialog.askopenfilename() )
    self.entry_file.focus_set()
    self.entry_file.selection_range(0, tkinter.END)

def Click(self):
    print(root.entry_file.get())

root = tkinter.Tk()
frame=tkinter.Frame(root)

firstButton = tkinter.Button(frame, text ="Choose", command=lambda:
OnDirectoryButtonClick(root))
firstButton.grid(column=1,row=1)
root.entryVariable_dir = tkinter.StringVar()
root.entry_dir = tkinter.Entry(frame,textvariable=root.entryVariable_dir)
root.entry_dir.grid(column=0,row=1)

secondButton = tkinter.Button(frame, text ="Choose", command=lambda: 
OnFileButtonClick(root))
secondButton.grid(column=1,row=3)
root.entryVariable_file = tkinter.StringVar()
root.entry_file = tkinter.Entry(frame,textvariable=root.entryVariable_file)
root.entry_file.grid(column=0,row=3)

secondButton = tkinter.Button(frame, text ="Choose", command=lambda: 
Click(root))
secondButton.grid(column=1,row=5)

print(root.entry_file.get())

frame.pack(pady=9)

root.mainloop()