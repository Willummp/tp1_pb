import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# Identificar as características (features) e o alvo (target)
features = ['Pclass', 'Sex', 'Age']
target = 'Survived'

# Transformar a coluna 'Sex' em valores numéricos
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})

# Tratar valores ausentes na coluna 'Age'
train['Age'].fillna(train['Age'].median(), inplace=True)

# Selecionar as features e o target
X = train[features]
y = train[target]

# Dividir os dados em conjuntos de treino e validação (80% treino, 20% validação)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar as variáveis dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)