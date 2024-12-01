import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'D:/Brain_Tumor_Detection/brain_tumor_model.pth'  # Update this with your model path
model = ConvNet(num_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define image preprocessing function
def predict_tumor_type(image):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Preprocess the image
    img = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img)
    
    # Print raw output and softmax probabilities
    print("Raw logits:", output)
    probabilities = torch.softmax(output, dim=1)
    print("Probabilities:", probabilities)
    
    # Get predicted class
    _, predicted = torch.max(probabilities, 1)
    classes = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    predicted_class = classes[predicted.item()]
    
    return predicted_class

# Test the model with a sample image
image_path = 'path_to_your_image.jpg'  # Change to the path of your test image
image = Image.open(image_path)
predicted_class = predict_tumor_type(image)
print(f"Predicted tumor type: {predicted_class}")
