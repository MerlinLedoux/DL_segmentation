import os
import torch
import numpy as np
import cv2
import argparse
from segmentation_models_pytorch import UnetPlusPlus
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from torch.nn.functional import softmax

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/best_model7517.pth"  # Chemin par défaut vers le modèle sauvegardé
OUTPUT_DIR = "./Output"  # Répertoire par défaut pour sauvegarder les masques prédits
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fonction de transformation des images
def preprocess_image(image_path, resize=(256, 256)):
    """Charge et transforme une image pour l'inférence."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"L'image {image_path} est introuvable.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize)
    
    # Normalisation et conversion en tenseur
    transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0)  # Ajoute une dimension batch

# Chargement du modèle
def load_model(model_path):
    """Charge un modèle UNet++ avec les poids sauvegardés."""
    model = UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None, 
        in_channels=3,
        classes=3
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Fonction de conversion du masque en RGB
def mask_to_rgb(mask, color_dict):
    """Convertit un masque de classes en une image RGB."""
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_dict.items():
        output[mask == class_id] = color
    return output

# Dictionnaire de couleurs pour la visualisation
COLOR_DICT = {
    0: (0, 0, 0),       # Classe 0 : Noir
    1: (255, 0, 0),     # Classe 1 : Rouge
    2: (0, 255, 0)      # Classe 2 : Vert
}

# Inférence sur une image
def infer_image(model, image_path, output_path):
    """Réalise l'inférence sur une image donnée et sauvegarde le masque prédit."""
    image_tensor = preprocess_image(image_path).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        output = softmax(output, dim=1)  # Probabilités par classe
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Classe avec la probabilité maximale

    # Conversion du masque en RGB
    mask_rgb = mask_to_rgb(mask, COLOR_DICT)
    cv2.imwrite(output_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))  # Sauvegarde du masque en BGR
    print(f"Masque sauvegardé dans {output_path}")

# Fonction principale avec argparse
def main():
    parser = argparse.ArgumentParser(description="Inférence pour la segmentation d'images.")
    parser.add_argument("--image_path", type=str, required=True, help="Chemin de l'image d'entrée.")
    parser.add_argument("--output_path", type=str, default=None, help="Chemin de sortie pour l'image prédite.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Chemin vers le modèle sauvegardé.")
    args = parser.parse_args()

    # Chargement du modèle
    model = load_model(args.model_path)

    # Inférence sur l'image donnée
    image_path = args.image_path
    output_path = args.output_path or os.path.join(OUTPUT_DIR, os.path.basename(image_path))
    infer_image(model, image_path, output_path)

if __name__ == "__main__":
    main()
