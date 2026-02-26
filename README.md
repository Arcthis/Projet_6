# Projet 6 - Prédiction de la consommation énergétique des bâtiments

## Contexte et objectifs

Ce projet s'inscrit dans une démarche de transition énergétique menée par la ville de Seattle. La ville a pour ambition d'atteindre la neutralité carbone à l'horizon 2050. Dans ce cadre, elle impose aux propriétaires de bâtiments non résidentiels de déclarer annuellement leur consommation énergétique et leurs émissions de CO2, conformément au programme "Building Energy Benchmarking".

L'objectif du projet est double :

1. **Prédire la consommation énergétique annuelle** (`SiteEnergyUse`) des bâtiments non résidentiels de Seattle à partir de leurs caractéristiques structurelles, en se passant des relevés de consommation directe qui sont coûteux à collecter.
2. **Déployer ce modèle de prédiction** sous forme d'une API REST accessible et conteneurisée, permettant d'interroger le modèle à la demande.

Le jeu de données utilisé est le relevé de benchmarking énergétique de Seattle pour l'année 2016, comprenant environ 3 376 bâtiments non résidentiels et 46 variables descriptives (surface, type de bâtiment, nombre d'étages, émissions de gaz à effet de serre, ancienneté, etc.).

---

## Technologies utilisées

### Analyse et modélisation
- **Python** : langage principal du projet
- **Pandas / NumPy** : manipulation et traitement des données
- **Matplotlib / Seaborn** : visualisation des données
- **Scikit-learn** : prétraitement des données, entraînement et évaluation des modèles de machine learning (régression linéaire, SVR, Random Forest)

### Déploiement de l'API
- **BentoML** : framework de déploiement de modèles de machine learning, utilisé pour packager et servir le modèle sous forme d'API REST
- **Pydantic** : validation des données d'entrée de l'API
- **Docker** : conteneurisation de l'application pour faciliter le déploiement

---

## Installation et utilisation

### Prérequis
- Python installé sur votre machine
- Docker installé (pour le déploiement conteneurisé)

### Lancer l'API en local (avec BentoML)

1. Installer les dépendances :
   ```bash
   pip install -r api/requirements.txt
   ```

2. Importer le modèle dans le registre BentoML (nécessite d'avoir exécuté le notebook au préalable pour générer le fichier `.pkl`), puis lancer le service :
   ```bash
   cd api
   bentoml serve service:svc
   ```

3. L'API est accessible sur `http://localhost:3000`.

### Lancer l'API avec Docker

1. Construire l'image Docker :
   ```bash
   docker build -t energy-prediction-api ./api
   ```

2. Lancer le conteneur :
   ```bash
   docker run -p 3000:3000 energy-prediction-api
   ```

### Exemple d'appel à l'API

```bash
curl -X POST http://localhost:3000/api/v1/energy \
  -H "Content-Type: application/json" \
  -d '{
        "BuildingAge": 90,
        "BuildingType": "NonResidential",
        "NumberofFloors": 5,
        "PropertyGFATotal": 15000,
        "PropertyGFAParking": 2000,
        "TotalGHGEmissions": 50
      }'
```

La réponse retourne la consommation énergétique prédite en kBtu :
```json
{"predicted_energy_use": 1234567.89}
```

---

## Résultats obtenus

### Démarche de réalisation

Le projet a suivi les grandes étapes classiques d'un projet de machine learning supervisé :

1. **Analyse exploratoire** : compréhension du jeu de données, identification des valeurs manquantes, analyse des distributions et des corrélations entre variables.
2. **Feature engineering** : création de nouvelles variables à partir des données existantes (ancienneté du bâtiment, ratio de surface de parking, surface moyenne par étage).
3. **Préparation des données** : traitement des valeurs aberrantes (outliers), encodage des variables catégorielles, standardisation des features.
4. **Entraînement et comparaison de modèles** : trois algorithmes ont été comparés — Régression Linéaire, SVR (Support Vector Regression) et Random Forest.
5. **Optimisation** : recherche des meilleurs hyperparamètres pour le Random Forest via GridSearchCV.
6. **Déploiement** : le modèle de régression linéaire a été retenu pour le déploiement en raison de son bon équilibre entre performance et interpretabilité. Il a été packagé avec BentoML et conteneurisé avec Docker.

### Performances des modèles

| Modèle                     | R² (test) | MAE (test)     |
|----------------------------|-----------|----------------|
| Régression Linéaire         | 0.74      | ~636 000 kBtu  |
| SVR                        | -0.14     | ~1 420 000 kBtu|
| Random Forest (optimisé)   | 0.83      | ~465 000 kBtu  |

Le **Random Forest optimisé** offre les meilleures performances prédictives (R² de 0.83 sur le jeu de test). La **Régression Linéaire** présente des résultats satisfaisants (R² de 0.74) avec une bien meilleure interpretabilité, ce qui justifie son choix pour le déploiement en production.

Les variables les plus influentes dans la prédiction sont les émissions totales de gaz à effet de serre (`TotalGHGEmissions`), la surface totale du bâtiment (`PropertyGFATotal`) et l'ancienneté du bâtiment (`BuildingAge`).