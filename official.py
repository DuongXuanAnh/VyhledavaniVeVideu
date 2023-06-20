import tkinter as tk
from tkinter import ttk
 
from PIL import Image, ImageTk
import os
 
shown = 96
dataset_path = "Images/"  # TODO: set path to dataset

# get images address
filenames = []
for fn in sorted(os.listdir(dataset_path)):
    filename = dataset_path + fn
    filenames.append(filename)
 
root = tk.Tk()
root.title("Searcher")
root.wm_attributes('-fullscreen', 'true')
 
image_size = (int(root.winfo_screenwidth() / 14) - 4, int(root.winfo_screenheight() / 8) - 6)
 
images_buttons = []
selected_images = []
shown_images = []
 
 
def hide_borders():
    global selected_images
    for button in images_buttons:
        button.config(bg="black")
    selected_images = []
 
 
def search_clip(text):
    print(text)
    top_result = ...  # TODO: text query clip search
    # top result - sorted score (position)
    for i in range(shown):
        shown_images[i] = ImageTk.PhotoImage(Image.open(filenames[top_result[i]]).resize(image_size))
        images_buttons[i].configure(image=shown_images[i], text=filenames[top_result[i]],
                                    command=(lambda j=i: on_click(j)))
    hide_borders()
 
 
def on_click(index):
    if images_buttons[index].cget("bg") == "yellow":
        images_buttons[index].config(bg="black")
        selected_images.remove(images_buttons[index].cget("text"))
    else:
        images_buttons[index].config(bg="yellow")
        selected_images.append(images_buttons[index].cget("text")) 
 
def on_double_click():
    hide_borders()
 
 
def close_win(e):
    root.destroy()
 
# create window
window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
window.pack(fill=tk.BOTH, expand=True)
 
# create frames
search_bar = ttk.Frame(window, width=root.winfo_screenwidth() / 4, height=root.winfo_screenheight(), relief=tk.SUNKEN)
result_frame = ttk.Frame(window, width=(3 * root.winfo_screenwidth()) / 4, height=root.winfo_screenheight(),
                         relief=tk.SUNKEN)
window.add(search_bar, weight=1)
window.add(result_frame, weight=4)
 
# add text input
tk.Label(search_bar, text="Text query:").pack(side=tk.TOP, pady=5)
text_input = tk.Entry(search_bar, bd=3, width=32)
text_input.bind("<Return>", (lambda event: search_clip(text_input.get())))
text_input.pack(side=tk.TOP, pady=5)
 
# add search buttons
clip_button = tk.Button(search_bar, text="Search Clip", command=(lambda: search_clip(text_input.get())))
clip_button.pack(side=tk.TOP)
 
# set images
for s in range(shown):
    # load image
    shown_images.append(ImageTk.PhotoImage(Image.open(filenames[s]).resize(image_size)))
    # create button
    images_buttons.append(tk.Button(result_frame, bg="black", bd=2, text=filenames[s], image=shown_images[s],
                                    command=(lambda j=s: on_click(j))))
    # set position of button
    images_buttons[s].grid(row=(s // 12), column=(s % 12), sticky=tk.W)
    # set double click to reset marking of images
    images_buttons[s].bind('<Double-1>', lambda event: on_double_click())
 
# set escape as exit
root.bind('<Escape>', lambda e: close_win(e))
 
root.mainloop()