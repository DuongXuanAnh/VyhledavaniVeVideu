import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import pandas as pd
import torch
import clip

dataset_path = ""
csv_file_path = "CLIP_VITB32.csv"

# get images address
filenames = []
shown_images = []
images_buttons = []
shown = 96

root = tk.Tk()
root.title("Searcher")
root.wm_attributes('-fullscreen', 'true')

image_size = (int(root.winfo_screenwidth() / 14) - 7, int(root.winfo_screenheight() / 8) - 6)

selected_images = []

def hide_borders():
    global selected_images
    for button in images_buttons:
        button.config(bg="black")
    selected_images = []

def cosine_distance(v1, v2): # cang nho cang giong
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def topSimilarImages(text, numberOfImages = 96):
    if (text == ""):
        return range(numberOfImages)
    else:
           # Check if CSV file exists
        if not os.path.isfile(csv_file_path):
            raise FileNotFoundError(csv_file_path + ' not found.')

        # Load the CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Tokenize and encode text
        text = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        text_vector = np.array(text_features[0], dtype=np.float32)

        # Load vectors in chunks to handle large files
        chunksize = 50000
        similarities = []
        for chunk in pd.read_csv(csv_file_path, sep=";", chunksize=chunksize):
            vectors = chunk.to_numpy()
            # Compute similarities using vectorized operations for efficiency
            similarities.extend([cosine_distance(text_vector, v) for v in vectors])

        # Create DataFrame from similarities
        df = pd.DataFrame(similarities, columns=['similarity'])

        # Sort by similarity
        df = df.sort_values(by='similarity', ascending=True)

        # Return indices of top similar images
        top_vectors = df[:numberOfImages]

        return top_vectors.index.to_list()
    
def topSimilarImagesUsingSimilarity(imgID, numberOfImages = 96):
    df = pd.read_csv(csv_file_path, sep=";")
    vectors = df.to_numpy()
    imgID = int(imgID)
    img_vector = vectors[imgID]
    similarities = [cosine_distance(img_vector, v) for v in vectors]
    # Add the similarities as a new column to the DataFrame
    df['similarity'] = similarities
    # Sort the vectors by their similarity to the reference vector
    df = df.sort_values(by='similarity', ascending=True)

    top_vectors = df[:numberOfImages]

    return top_vectors.index.to_list()
   

def search_clip(text):
    top_result = topSimilarImages(text, numberOfImages = shown)  # replace this line with your search function
    for i in range(shown):
        image = Image.open(filenames[top_result[i]]).resize(image_size)
        photo = ImageTk.PhotoImage(image)
        shown_images.append(photo)  # update the shown_images list
        images_buttons[i].configure(image=photo, text=filenames[top_result[i]], command=(lambda j=i: on_click(j)))
    hide_borders()

def on_click(index):
    if images_buttons[index].cget("bg") == "yellow":
        images_buttons[index].config(bg="black")
        selected_images.remove(images_buttons[index].cget("text"))
    else:
        images_buttons[index].config(bg="yellow")
        selected_images.append(images_buttons[index].cget("text"))
    text_index.config(text="Last selected image: " + os.path.basename(selected_images[0]))

def on_double_click():
    hide_borders()

def close_win(e):
    root.destroy()


def select_directory():
    global dataset_path
    dataset_path = filedialog.askdirectory()

    # -------------------------------------------------------------------------

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device)

    # try:
    #     with open(csv_file_path, 'w') as csv_file:
    #         file_list = os.listdir(dataset_path)

    #         for filename in file_list:
    #             image_path = os.path.join(dataset_path, filename)
    #             try:
    #                 image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    #                 image_vector = model.encode_image(image)
    #                 image_vector = image_vector / image_vector.norm(dim=-1, keepdim=True)
    #                 image_vector_str = ';'.join(str(value) for value in image_vector.flatten().tolist())
    #                 csv_file.write(image_vector_str + '\n')
    #             except Exception:
    #                 print(f"Skipping {image_path} as it is not a valid image file.")
    #     load_images_from_directory(dataset_path)

    #     messagebox.showinfo("Loading Complete", "Images have been loaded successfully.")
    # except Exception as e:
    #     messagebox.showerror("Loading Error", f"An error occurred while loading images:\n\n{str(e)}")
    # -------------------------------------------------------------------------
    load_images_from_directory(dataset_path)


def load_images_from_directory(directory):
    global filenames, shown_images, shown, images_buttons
    # Clear all existing data
    filenames = []
    shown_images = []
    for btn in images_buttons:
        btn.grid_forget()  # This removes the button from grid
    images_buttons = []
    if os.path.exists(directory):
        for fn in sorted(os.listdir(directory)):
            filename = os.path.join(directory, fn)
            filenames.append(os.path.normpath(filename))
    shown = min(96, len(filenames))
    create_buttons()  # create the buttons here
    if shown > 0:
        search_clip('')


def find_similar_pictures():
    imgID = filenames.index(os.path.normpath(selected_images[0]))
    top_result = topSimilarImagesUsingSimilarity(imgID, numberOfImages = shown)  # replace this line with your search function
    for i in range(shown):
        image = Image.open(filenames[top_result[i]]).resize(image_size)
        photo = ImageTk.PhotoImage(image)
        shown_images.append(photo)  # update the shown_images list
        images_buttons[i].configure(image=photo, text=filenames[top_result[i]], command=(lambda j=i: on_click(j)))
    hide_borders()


def find_back_img():
    imgID = int(filenames.index(os.path.normpath(selected_images[0])))
    top_result = []
    for i in range(shown):
        if(imgID == (len(filenames) - 1)):
            top_result.append(imgID)
            imgID = 0 - i
        top_result.append(i + imgID)
 
    for i in range(shown):
        shown_images[i] = ImageTk.PhotoImage(Image.open(filenames[top_result[i]]).resize(image_size))
        images_buttons[i].configure(image=shown_images[i], text=filenames[top_result[i]],
                                    command=(lambda j=i: on_click(j)))
    hide_borders()

def find_front_img():
    imgID = int(filenames.index(os.path.normpath(selected_images[0])))
    top_result = []
    for i in range(96):
        top_result.append(imgID - i)
    
    for i in range(shown):
        shown_images[i] = ImageTk.PhotoImage(Image.open(filenames[top_result[shown - i - 1]]).resize(image_size))
        images_buttons[i].configure(image=shown_images[i], text=filenames[top_result[shown - i - 1]],
                                    command=(lambda j=i: on_click(j)))
    hide_borders()


def create_buttons():
    for s in range(shown):
        # create button
        btn = tk.Button(result_frame, bg="black", bd=2, command=None)
        images_buttons.append(btn)
        # set position of button
        btn.grid(row=(s // 12), column=(s % 12), sticky=tk.W)
        # set double click to reset marking of images
        btn.bind('<Double-1>', lambda event: on_double_click())
    for s in range(shown, len(images_buttons)):  # remove any extra buttons
        images_buttons[s].grid_forget()

# create window
window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
window.pack(fill=tk.BOTH, expand=True)

# create frames
search_bar = ttk.Frame(window, width=root.winfo_screenwidth() / 4, height=root.winfo_screenheight(), relief=tk.SUNKEN)
result_frame = ttk.Frame(window, width=(3 * root.winfo_screenwidth()) / 4, height=root.winfo_screenheight(),
                         relief=tk.SUNKEN)
window.add(search_bar, weight=1)
window.add(result_frame, weight=4)

# add directory selection button
dir_button = tk.Button(search_bar, text="Choose Directory", command=select_directory)
dir_button.pack(side=tk.TOP, pady=10)

# add text input
tk.Label(search_bar, text="Text query:").pack(side=tk.TOP, pady=5)
text_input = tk.Entry(search_bar, bd=3, width=32)
text_input.bind("<Return>", (lambda event: search_clip(text_input.get())))
text_input.pack(side=tk.TOP, pady=5)

# add search buttons
clip_button = tk.Button(search_bar, text="Search Clip", command=(lambda: search_clip(text_input.get())))
clip_button.pack(side=tk.TOP)

# add info labels
text_index = tk.Label(search_bar, text="Last selected image: ")
text_index.pack(side=tk.TOP, pady=5)

# Find similar pictures
find_similar_img_b = tk.Button(search_bar, text="Find similar pictures", command=(lambda: find_similar_pictures()))
find_similar_img_b.pack(side=tk.TOP, pady=100)


# Create a frame to contain the navigation buttons
navigation_frame = ttk.Frame(search_bar)
navigation_frame.pack(side=tk.TOP, pady=5)

# Find front pictures
find_front_img_b = tk.Button(navigation_frame, text="Prev", command=(lambda: find_front_img()))
find_front_img_b.pack(side=tk.LEFT)

# Find back pictures
find_back_img_b = tk.Button(navigation_frame, text="Next", command=(lambda: find_back_img()))
find_back_img_b.pack(side=tk.LEFT)



# set escape as exit
root.bind('<Escape>', lambda e: close_win(e))

root.mainloop()