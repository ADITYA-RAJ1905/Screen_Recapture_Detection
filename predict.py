import sys
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from skimage.feature import local_binary_pattern


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_effnet():
    """Load the trained EfficientNet model."""

    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )

    model.load_state_dict(
        torch.load("enet.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    return model


def load_xgboost():
    return joblib.load("xgboost.pkl")


effnet = load_effnet()
xgb = load_xgboost()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_features(image_path):
    

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Unable to open image: {image_path}"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    feature_vector = []

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    feature_vector.append(sharpness)

    edges = cv2.Canny(gray, 100, 200)

    edge_ratio = np.count_nonzero(edges) / edges.size

    feature_vector.append(edge_ratio)

    glare_ratio = np.sum(gray > 240) / gray.size

    feature_vector.append(glare_ratio)

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(spectrum)

    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    radius = min(h, w) // 8

    mask = np.ones_like(magnitude)

    mask[
        center_y - radius:center_y + radius,
        center_x - radius:center_x + radius
    ] = 0

    high_frequency_energy = np.mean(magnitude * mask)

    feature_vector.append(high_frequency_energy)

    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        density=True
    )

    feature_vector.extend(histogram)

    return np.array(feature_vector)


def predict(image_path):
  

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = effnet(input_tensor)

        cnn_score = torch.softmax(
            output,
            dim=1
        )[0][1].item()

    handcrafted_features = extract_features(image_path)
    handcrafted_features = handcrafted_features.reshape(1, -1)

    xgb_score = xgb.predict_proba(
        handcrafted_features
    )[0][1]

    final_score = 0.6 * cnn_score + 0.4 * xgb_score

    return final_score


def main():

    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    score = predict(image_path)

    print(f"{score:.4f}")


if __name__ == "__main__":
    main()