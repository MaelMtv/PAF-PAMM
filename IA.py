import cv2
import numpy as np
import pandas as pd
import glob


def extract_features(image):  
    # Réduire le bruit de l'image en appliquant un flou gaussien
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Appliquer la détection de contours pour extraire les contours de l'image
    edges = cv2.Canny(blurred_image, 50, 150)
    
    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculer les caractéristiques géométriques des contours
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        area = 0
        perimeter = 0
        circularity = 0
    
    # Calculer les histogrammes de couleur pour obtenir des caractéristiques de couleur
    hist_b = cv2.calcHist([image], [0], None, [25], [0, 25])
    hist_g = cv2.calcHist([image], [1], None, [25], [0, 25])
    hist_r = cv2.calcHist([image], [2], None, [25], [0, 25])
    
    # Concaténer toutes les caractéristiques extraites en un vecteur
    features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten(), [area, perimeter, circularity]])
    
    return features




# Chemin vers le dossier contenant les images
dossier_images_2016 = "C:/Users/Alexandre/Downloads/archive2016/*.jpg"  # Modifier le chemin et le motif selon vos besoins
dossier_images_2017 = "C:/Users/Alexandre/Downloads/archive2017/*.jpg"  # Modifier le chemin et le motif selon vos besoins
dossier_malignant_2016 = "C:/Users/Alexandre/Downloads/malignant_2016/*.jpg"  # Modifier le chemin et le motif selon vos besoins
kaggle_benign = "C:/Users/Alexandre/Downloads/kaggle/NotMelanoma/*.jpg"  # Modifier le chemin et le motif selon vos besoins
kaggle_malignant = "C:/Users/Alexandre/Downloads/kaggle/Melanoma/*.jpg"  # Modifier le chemin et le motif selon vos besoins

# Liste pour stocker les images chargées
images = []

data_2016 = pd.read_csv("C:/Users/Alexandre/Downloads/database2016.csv", sep=';')
data_2017 = pd.read_csv("C:/Users/Alexandre/Downloads/database2017.csv", sep=';')
data_malignant_2016 = pd.read_csv("C:/Users/Alexandre/Downloads/malignant_training_2016.csv", sep=';')
data_kaggle_malignant = pd.read_csv("C:/Users/Alexandre/Downloads/kaggle_malignant.csv", sep=';')
data_kaggle_benign = pd.read_csv("C:/Users/Alexandre/Downloads/kaggle_benign.csv", sep=';')

mapping_2016 = dict(zip(data_2016['isic_id'], data_2016['benign_malignant']))
mapping_2017 = dict(zip(data_2017['isic_id'], data_2017['benign_malignant']))
mapping_malignant_2016 = dict(zip(data_malignant_2016['isic_id'], data_malignant_2016['benign_malignant']))
mapping_kaggle_malignant = dict(zip(data_kaggle_malignant['isic_id'], data_kaggle_malignant['benign_malignant']))
mapping_kaggle_benign = dict(zip(data_kaggle_benign['isic_id'], data_kaggle_benign['benign_malignant']))
mentions_images = []

X = []
y = []

for fichier_image in glob.glob(dossier_images_2016):
    # Charger l'image
    image = cv2.imread(fichier_image)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Vérifier si l'image a été chargée avec succès
    if image is not None:
        images.append(image)
    else:
        print("Erreur lors du chargement de l'image:", fichier_image)

    caracteristiques = extract_features(image)
    # Obtenir l'identifiant de l'image
    identifiant_image = int(fichier_image.split("\\")[-1].split(".")[0])
    # Récupérer la mention correspondante
    mention = mapping_2016.get(identifiant_image)
    if mention is not None:
        mentions_images.append(mention)
        X.append(caracteristiques)
        y.append(mention)
    else:
       print("Mention non trouvée pour l'image:", fichier_image)


for fichier_image in glob.glob(dossier_images_2017):
    # Charger l'image
    image = cv2.imread(fichier_image)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Vérifier si l'image a été chargée avec succès
    if image is not None:
        images.append(image)
    else:
        print("Erreur lors du chargement de l'image:", fichier_image)

    caracteristiques = extract_features(image)
    # Obtenir l'identifiant de l'image
    identifiant_image = int(fichier_image.split("\\")[-1].split(".")[0])
    # Récupérer la mention correspondante
    mention = mapping_2017.get(identifiant_image)
    if mention is not None:
        mentions_images.append(mention)
        X.append(caracteristiques)
        y.append(mention)
    else:
       print("Mention non trouvée pour l'image:", fichier_image)


for fichier_image in glob.glob(dossier_malignant_2016):
    # Charger l'image
    image = cv2.imread(fichier_image)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Vérifier si l'image a été chargée avec succès
    if image is not None:
        images.append(image)
    else:
        print("Erreur lors du chargement de l'image:", fichier_image)

    caracteristiques = extract_features(image)
    # Obtenir l'identifiant de l'image
    identifiant_image = int(fichier_image.split("\\")[-1].split(".")[0])
    # Récupérer la mention correspondante
    mention = mapping_malignant_2016.get(identifiant_image)
    if mention is not None:
        mentions_images.append(mention)
        X.append(caracteristiques)
        y.append(mention)
    else:
       print("Mention non trouvée pour l'image:", fichier_image)


for fichier_image in glob.glob(kaggle_malignant):
    # Charger l'image
    image = cv2.imread(fichier_image)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Vérifier si l'image a été chargée avec succès
    if image is not None:
        images.append(image)
    else:
        print("Erreur lors du chargement de l'image:", fichier_image)

    caracteristiques = extract_features(image)
    # Obtenir l'identifiant de l'image
    identifiant_image = int(fichier_image.split("\\")[-1].split(".")[0])
    # Récupérer la mention correspondante
    mention = mapping_kaggle_malignant.get(identifiant_image)
    if mention is not None:
        mentions_images.append(mention)
        X.append(caracteristiques)
        y.append(mention)
    else:
       print("Mention non trouvée pour l'image:", fichier_image)


for fichier_image in glob.glob(kaggle_benign):
    # Charger l'image
    image = cv2.imread(fichier_image)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Vérifier si l'image a été chargée avec succès
    if image is not None:
        images.append(image)
    else:
        print("Erreur lors du chargement de l'image:", fichier_image)

    caracteristiques = extract_features(image)
    # Obtenir l'identifiant de l'image
    identifiant_image = int(fichier_image.split("\\")[-1].split(".")[0])
    # Récupérer la mention correspondante
    mention = mapping_kaggle_benign.get(identifiant_image)
    if mention is not None:
        mentions_images.append(mention)
        X.append(caracteristiques)
        y.append(mention)
    else:
       print("Mention non trouvée pour l'image:", fichier_image)

# Afficher le nombre d'images chargées
print("Nombre d'images chargées:", len(images))


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Préparez vos données d'entraînement X et les étiquettes y correspondantes

# Initialisez le modèle RandomForestClassifier
model = RandomForestClassifier()
#model= SVC(kernel='linear')
X=np.array(X)
y=np.array(y)
# Entraînez le modèle sur vos données d'entraînement
model.fit(X, y)


dossier_test = "C:/Users/Alexandre/Downloads/TestIA/*.jpg"
for fichier_image_test in glob.glob(dossier_test):
    # Charger l'image
    nouvelle_image = cv2.imread(fichier_image_test)
    #nouvelle_image_gray = cv2.cvtColor(nouvelle_image, cv2.COLOR_BGR2GRAY)
    # Extraire les caractéristiques de l'image prétraitée
    caracteristiques = extract_features(nouvelle_image)

    # Utiliser le modèle ou l'algorithme entraîné pour prédire la mention
    #prediction = model.predict(np.array([caracteristiques]))
    prediction = model.predict([caracteristiques])
    # Interpréter la prédiction
    if prediction[0] == 'benign':
     mention_predite = "benign"
    else:
        mention_predite = 'malignant'

    # Afficher la mention prédite
    print("La mention prédite pour l'image",fichier_image_test.split("\\")[-1].split(".")[0], "est :", mention_predite)

# Charger et prétraiter la nouvelle image
#nouvelle_image = cv2.imread("C:/Users/Alexandre/Downloads/archiveTest/Test1.jpg")
#nouvelle_image_gray = cv2.cvtColor(nouvelle_image, cv2.COLOR_BGR2GRAY)

