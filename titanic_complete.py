# Titanic Survival Analysis - Complete Day 1 Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🚀 STARTING DAY 1: TITANIC SURVIVAL ANALYSIS")
print("="*50)

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("🔍 Dataset Shape:", df.shape)
print("📋 Columns:", df.columns.tolist())

# Data cleaning
print("\n🛠️ Data Cleaning...")
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# Basic insights
print("\n📊 Basic Insights:")
print(f"Overall survival rate: {df['Survived'].mean()*100:.1f}%")
print(f"Female survival rate: {df[df['Sex']=='female']['Survived'].mean()*100:.1f}%")
print(f"Male survival rate: {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")
print(f"1st Class survival: {df[df['Pclass']==1]['Survived'].mean()*100:.1f}%")
print(f"3rd Class survival: {df[df['Pclass']==3]['Survived'].mean()*100:.1f}%")

# Prepare data for ML
print("\n🤖 Preparing Machine Learning Model...")
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked_encoded'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy: {accuracy:.2%}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Feature Importance Ranking:")
for i, row in feature_importance.iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")

# Final Summary
print("\n" + "="*60)
print("🎭 DAY 1 COMPLETE: TITANIC DATA STORY")
print("="*60)

print("\n📖 KEY FINDINGS:")
print(f"• Survival rate: {df['Survived'].mean()*100:.1f}%")
print(f"• Women had {df[df['Sex']=='female']['Survived'].mean()*100:.1f}% survival vs Men {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")
print(f"• 1st Class: {df[df['Pclass']==1]['Survived'].mean()*100:.1f}% vs 3rd Class: {df[df['Pclass']==3]['Survived'].mean()*100:.1f}%")
print(f"• AI Prediction Accuracy: {accuracy:.2%}")
print(f"• Most Important Factor: {feature_importance['feature'].iloc[0]}")

print("\n✅ SUCCESSFULLY COMPLETED DAY 1!")
print("🚀 Tomorrow: Netflix Content Strategy Analysis!")

print("\n" + "="*50)
print("Ready for Day 2? Let me know when you want to continue!")
print("="*50)
