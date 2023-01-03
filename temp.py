import tkinter as tk
from tkinter import ttk
import requests
from PIL import Image, ImageTk
import os
import numpy as np
import pandas as pd
import torch
import clip
import requests
 
shown = 96
url = "https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php"
dataset_path = "Images/"


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

# ---------------------------------------------------------------------------------------------------

def cosine_distance(v1, v2): # cang nho cang giong
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))

def pearson_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)
    cov = np.sum((x - x_mean) * (y - y_mean))
    return 0 - cov / (x_std * y_std)

def jaccard_similarity(vec1, vec2): # Hledani coralu (San ho)
  intersection = set(vec1).intersection(vec2)
  union = set(vec1).union(vec2)
  return len(intersection) / len(union)

def topSimilarImages(text, numberOfImages = 10):
    # Z textu udela vektor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

    text_vector = np.array(text_features[0], dtype=np.float32)

    df = pd.read_csv('CLIP_VITB32.csv', sep=";")

    vectors = df.to_numpy()

    similarities = [cosine_distance(text_vector, v) for v in vectors]

    # Add the similarities as a new column to the DataFrame
    df['similarity'] = similarities

    # Sort the vectors by their similarity to the reference vector
    df = df.sort_values(by='similarity', ascending=True)

    # Select the top 5 vectors
    top_vectors = df[:numberOfImages]

    return top_vectors.index.to_list()

# ----------------------------------------------------------------------------------------------------

def search_clip(text):
    # print(text)
    top_result = topSimilarImages(text, numberOfImages = shown) # TODO: text query clip search
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
    text_index.config(text="Last selected image: " + selected_images[0][-9:])
 
 
def on_double_click():
    hide_borders()
 
 
def close_win(e):
    root.destroy()
 
 
def send_result():
    key_i = (selected_images[0][-9:])[:5]

    print(key_i)

    my_obj = {'team': "duongx", 'item': key_i}
 
    x = requests.get(url=url, params=my_obj, verify=False)
    print(x.text)

def find_similar_pictures():
    key_i = (selected_images[0][-9:])[:5]
    print(key_i)
 
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
 
# add info labels
tk.Label(search_bar, text="Find index (1-11870):").pack(side=tk.TOP, pady=10)
text_index = tk.Label(search_bar, text="Last selected image: ")
text_index.pack(side=tk.TOP, pady=5)
 
# sending select result
send_result_b = tk.Button(search_bar, text="Send selected index", command=(lambda: send_result()))
send_result_b.pack(side=tk.TOP, pady=5)


# Find similar pictures
find_similar_img_b = tk.Button(search_bar, text="Find similar pictures", command=(lambda: find_similar_pictures()))
find_similar_img_b.pack(side=tk.TOP, pady=5)
# set control-v to set result
root.bind('<Control-v>', lambda e: send_result())
 
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
