import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi
import io
import base64

app = Flask(__name__)
CORS(app)

# Wrap Flask (WSGI) app as ASGI so servers like Uvicorn can run it
asgi_app = WsgiToAsgi(app)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the model
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    
    model_path = os.path.join(os.path.dirname(__file__), 'best_model', 'ResNet.pth')
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove 'module.' prefix if it exists
    if any(key.startswith("module.") for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
# Initialize the model
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the base64 encoded image from the request
        image_data = request.json['image']
        # Remove the data URL prefix if present
        if 'data:image' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get the predicted class and probability
        predicted_class = class_names[predicted.item()]
        confidence = probabilities[predicted.item()].item()
        
        # Get all class probabilities
        all_probs = {class_names[i]: float(probabilities[i].item()) for i in range(len(class_names))}
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': all_probs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'model': 'ResNet18',
        'classes': class_names,
        'status': 'active'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)