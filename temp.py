import os
from PIL import Image
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = r"C:\Users\david\Pictures\Screenshots"  # Use raw string (r"") or double backslashes (\\)
file_path = 'data.csv'

with open(file_path, 'w') as csv_file:
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image_vector = model.encode_image(image)
            image_vector = image_vector / image_vector.norm(dim=-1, keepdim=True)
            image_vector_str = ';'.join(str(value) for value in image_vector.flatten().tolist())
            csv_file.write(image_vector_str + '\n')
        except Exception:
            print(f"Skipping {image_path} as it is not a valid image file.")

print('Data written to', file_path)
