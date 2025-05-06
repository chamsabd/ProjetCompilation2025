# 🤖 Automate Visualizer

Une application interactive en **Python + Streamlit** permettant de **créer, visualiser et minimiser** des automates finis.

## 🚀 Fonctionnalités

- ✅ Création d’un automate à partir :
  - d’une **expression régulière**
  - ou **manuellement** via une **table de transition**
- 🔁 Conversion automatique :
  - **ε-AFN → AFN → AFD → AFD Minimisé**
- 📊 Affichage des **tables de transitions** :
  - en format **liste**
  - ou **matrice par état et symbole**
- 📈 Visualisation graphique des automates via **Graphviz**
- 🧪 Interface utilisateur simple avec **Streamlit**

## 🖼️ Exemple d’automate

- **Entrée :** `(a|b)*abb`
- **Sortie :**
  - Table de transitions
  - Graphe de l'automate
  - AFD minimisé

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-utilisateur/automate-visualizer.git
cd automate-visualizer
```

### 2.  Installer les dépendances
**Dépendances principales :**
- streamlit
- graphviz
- pandas
### 3.  Lancer l’application
Une fois les dépendances installées, lance l'application avec la commande suivante :

```bash
streamlit run file.py
```
## 🙌 Auteurs

- 👩‍💻 Chams Abdelwahed
- 🏫 Projet académique –Eniso 2025

