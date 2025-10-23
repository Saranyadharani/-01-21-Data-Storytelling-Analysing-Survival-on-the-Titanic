# Essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Let's take our first look at the data
print("🔍 Dataset Shape:", df.shape)
print("\n📋 First 5 rows:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Basic Statistics:")
print(df.describe())

# Check for missing values
print("❓ Missing Values:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# Visualize missing values
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Let's start our story with survival rates
print("🎯 Survival Overview:")
survival_count = df['Survived'].value_counts()
survival_rate = df['Survived'].value_counts(normalize=True) * 100

print(f"Survived: {survival_count[1]} ({survival_rate[1]:.1f}%)")
print(f"Did not survive: {survival_count[0]} ({survival_rate[0]:.1f}%)")

# Visualize survival
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='Survived', data=df, palette=['#ff6b6b', '#51cf66'])
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
df['Survived'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff6b6b', '#51cf66'])
plt.title('Survival Distribution')
plt.ylabel('')

plt.tight_layout()
plt.show()

# Handle missing values
print("🛠️ Handling Missing Values:")

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode (most common value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Cabin has too many missing values - we'll drop it for now
df.drop('Cabin', axis=1, inplace=True)

print("✅ Missing values handled!")
print(f"Remaining missing values: {df.isnull().sum().sum()}")

# 1. Survival by Gender - The most famous Titanic story!
print("👩‍👨 Survival by Gender:")
gender_survival = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
print(gender_survival)

plt.figure(figsize=(15, 10))

# Plot 1: Survival by Gender
plt.subplot(2, 3, 1)
gender_survival_plot = pd.crosstab(df['Sex'], df['Survived'])
gender_survival_plot.plot(kind='bar', color=['#ff6b6b', '#51cf66'])
plt.title('Survival by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['Did Not Survive', 'Survived'])
plt.xticks(rotation=0)

# 2. Survival by Passenger Class
plt.subplot(2, 3, 2)
class_survival = pd.crosstab(df['Pclass'], df['Survived'])
class_survival.plot(kind='bar', color=['#ff6b6b', '#51cf66'])
plt.title('Survival by Passenger Class')
plt.xlabel('Class (1 = First, 3 = Third)')
plt.ylabel('Count')
plt.legend(['Did Not Survive', 'Survived'])
plt.xticks(rotation=0)

# 3. Survival by Age Groups
plt.subplot(2, 3, 3)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
age_survival = pd.crosstab(df['AgeGroup'], df['Survived'], normalize='index') * 100
age_survival.plot(kind='bar', color=['#ff6b6b', '#51cf66'])
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate (%)')
plt.legend(['Did Not Survive', 'Survived'])
plt.xticks(rotation=45)

# 4. Fare distribution by survival
plt.subplot(2, 3, 4)
plt.hist([df[df['Survived'] == 0]['Fare'], df[df['Survived'] == 1]['Fare']], 
         bins=20, alpha=0.7, color=['#ff6b6b', '#51cf66'], label=['Did Not Survive', 'Survived'])
plt.title('Fare Distribution by Survival')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.legend()

# 5. Family size impact
plt.subplot(2, 3, 5)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
family_survival = df.groupby('FamilySize')['Survived'].mean() * 100
plt.plot(family_survival.index, family_survival.values, marker='o', color='#339af0')
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate (%)')
plt.grid(True)

# 6. Embarkation port survival
plt.subplot(2, 3, 6)
embark_survival = pd.crosstab(df['Embarked'], df['Survived'], normalize='index') * 100
embark_survival.plot(kind='bar', color=['#ff6b6b', '#51cf66'])
plt.title('Survival by Embarkation Port')
plt.xlabel('Port (C=Cherbourg, Q=Queenstown, S=Southampton)')
plt.ylabel('Survival Rate (%)')
plt.legend(['Did Not Survive', 'Survived'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Calculate key statistics for our story
print("📖 KEY INSIGHTS FROM TITANIC DATA STORY:")

# Women and children first?
female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
male_survival = df[df['Sex'] == 'male']['Survived'].mean() * 100
child_survival = df[df['Age'] < 18]['Survived'].mean() * 100

print(f"👩 Women survival rate: {female_survival:.1f}%")
print(f"👨 Men survival rate: {male_survival:.1f}%")
print(f"🧒 Children survival rate: {child_survival:.1f}%")

# Class matters?
for pclass in [1, 2, 3]:
    survival_rate = df[df['Pclass'] == pclass]['Survived'].mean() * 100
    print(f"🎫 Class {pclass} survival rate: {survival_rate:.1f}%")

# Money talks?
rich_survival = df[df['Fare'] > df['Fare'].median()]['Survived'].mean() * 100
poor_survival = df[df['Fare'] <= df['Fare'].median()]['Survived'].mean() * 100
print(f"💰 Higher fare survival rate: {rich_survival:.1f}%")
print(f"💸 Lower fare survival rate: {poor_survival:.1f}%")

# 🚀 STEP 7: BUILDING A PREDICTION MODEL
print("\n🤖 BUILDING MACHINE LEARNING MODEL...")

# Feature engineering for better predictions - FIXED THE ESCAPE SEQUENCE
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                    'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Convert categorical variables to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Select features for modeling
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.2%}")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Feature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance in Predicting Survival')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# 🎉 FINAL STORYTELLING SUMMARY
print("\n" + "="*60)
print("🎭 TITANIC DATA STORY: COMPLETE SUMMARY")
print("="*60)

print("\n📖 THE STORY REVEALED:")
print(f"• Overall survival rate: {df['Survived'].mean()*100:.1f}%")
print(f"• 'Women & children first' was REAL: Women {female_survival:.1f}% vs Men {male_survival:.1f}%")
print(f"• Class privilege: 1st Class {df[df['Pclass']==1]['Survived'].mean()*100:.1f}% vs 3rd Class {df[df['Pclass']==3]['Survived'].mean()*100:.1f}%")
print(f"• Wealth mattered: Higher fare passengers had {rich_survival:.1f}% survival rate")
print(f"• Our AI model can predict survival with {accuracy:.2%} accuracy")
print(f"• Most important survival factors: {feature_importance['feature'].iloc[0]} and {feature_importance['feature'].iloc[1]}")

print("\n🎯 DAY 1 COMPLETE! You've successfully:")
print("✅ Performed comprehensive data analysis")
print("✅ Created compelling visualizations") 
print("✅ Built a predictive ML model")
print("✅ Extracted meaningful business insights")
print("✅ Told a complete data story!")

# Continue with data cleaning and ML model
print("\n🛠️ Handling Missing Values...")

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

print("✅ Missing values handled!")

# Basic survival analysis
print("\n🎯 Survival Overview:")
survival_rate = df['Survived'].mean() * 100
print(f"Overall survival rate: {survival_rate:.1f}%")

# 🚀 BUILDING MACHINE LEARNING MODEL
print("\n" + "="*50)
print("🤖 BUILDING PREDICTION MODEL")
print("="*50)

# Simple feature engineering - FIXED VERSION
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked_encoded'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Select features for modeling
features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
X = df[features]
y = df['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Train Random Forest model
print("\n🏋️ Training Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy:.2%}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Most Important Survival Factors:")
for i, row in feature_importance.iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")

# 🎉 FINAL SUMMARY
print("\n" + "="*60)
print("🎭 TITANIC DATA STORY: COMPLETE SUMMARY")
print("="*60)

# Key insights
female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
male_survival = df[df['Sex'] == 'male']['Survived'].mean() * 100
class1_survival = df[df['Pclass'] == 1]['Survived'].mean() * 100
class3_survival = df[df['Pclass'] == 3]['Survived'].mean() * 100

print("\n📖 THE STORY REVEALED:")
print(f"• Overall survival: {survival_rate:.1f}%")
print(f"• Gender gap: Women {female_survival:.1f}% vs Men {male_survival:.1f}%")
print(f"• Class divide: 1st Class {class1_survival:.1f}% vs 3rd Class {class3_survival:.1f}%")
print(f"• AI prediction accuracy: {accuracy:.2%}")
print(f"• Key factor: {feature_importance['feature'].iloc[0]}")

print("\n✅ DAY 1 COMPLETED SUCCESSFULLY!")
