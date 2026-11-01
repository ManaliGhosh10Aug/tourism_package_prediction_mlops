# Tourism Package Prediction - MLOps Project

This repository implements the Great Learning Tourism Package Prediction project.

## Colab source path
Upload the dataset as:

`/content/tourism.csv`

The notebook then registers it in the repository at:

`data/tourism.csv`

## Repository workflow
1. Data validation
2. Data cleaning and train/test split
3. Train/test handoff using a GitHub Actions artifact
4. Random Forest tuning with GridSearchCV
5. MLflow experiment tracking
6. Model evaluation
7. Save best model
8. Streamlit deployment
9. GitHub Actions automatically commits the latest model to `main`

## Deployment
Streamlit entry point: `app.py`

## Required final evidence
- Public GitHub repository URL
- Screenshot of repository folder structure
- Screenshot of successful GitHub Actions workflow
- Public Streamlit Community Cloud URL
- Screenshot of working Streamlit app

Final Great Learning submission: exported HTML notebook.
