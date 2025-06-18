# ⚡ Système Intelligent de Relevé et de Facturation Électrique à partir d’Image de Compteur

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-red)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 📌 Description du Projet

Ce projet propose une solution intelligente pour automatiser le relevé d’index électrique à partir d’une simple photo d’un compteur. Grâce à l’intelligence artificielle (YOLOv8 + CNN + OCR), le système :

- 🧠 Détecte les zones importantes de l'image (numéro de compteur, index kWh)
- 🔢 Reconnaît les chiffres affichés sur les afficheurs 7 segments
- 📄 Génère un récapitulatif ou une facture automatiquement

---

## 🎯 Objectifs

- **Gain de temps** : éliminer la saisie manuelle
- **Réduction d’erreurs** : index mal lus ou mal entrés
- **Archivage digital** : traitement, enregistrement, export
- **Précision & performance** : modèles entraînés sur des jeux réels

---

## 🧠 Technologies utilisées

| Outil / Librairie     | Usage |
|------------------------|-------|
| `Flask`                | Application web (backend) |
| `YOLOv8 (Ultralytics)` | Détection des zones `index`, `compteur` |
| `CNN` (PyTorch)        | Reconnaissance des chiffres |
| `EasyOCR`, `Tesseract` | Lecture de texte brut (numéros de compteur et index) |
| `OpenCV`               | Traitement d’image |
| `FPDF` *(optionnel)*   | Génération de facture PDF |
| `HTML/CSS`             | Interface utilisateur |

---


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
