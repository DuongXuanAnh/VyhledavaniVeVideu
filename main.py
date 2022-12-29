import os
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Custom dataset class for the images in the "Images" folder
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Create a transform function to resize and normalize the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create an instance of the custom dataset class
dataset = CustomDataset("Images", transform=transform)

# Tokenize the text query
text_query = clip.tokenize("a photo of a cat")

# Calculate the image features and text features
image_features = []
text_features = []
for image in dataset:
    # Process the image
    image = preprocess(image).unsqueeze(0).to(device)
    # Calculate the image features
    image_features.append(model.encode_image(image))
    # Calculate the text features
    text_features.append(model.encode_text(text_query))

# Normalize the image features and text features
image_features = torch.stack(image_features)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features = torch.stack(text_features)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Calculate the similarity between the image
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Sort the images by their similarity scores in descending order
_, indices = similarity.sort(dim=-1, descending=True)

# Pick the top N images
N = 10
top_indices = indices[:N]

# Print the result
print("\nTop images:\n")
for i, index in enumerate(top_indices):
    image_name = dataset.image_names[index]
    similarity_score = 100 * similarity[i][index].item()
    print(f"{i+1}. {image_name}: {similarity_score:.2f}%")