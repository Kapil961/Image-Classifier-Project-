import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

checkpoint = torch.load("models/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

predicted_class = classes[predicted.item()]

print(f"Predicted Class: {predicted_class}")