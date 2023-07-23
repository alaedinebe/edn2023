import numpy as np
from sklearn.linear_model import LogisticRegression

# Supposons que 'data' est notre tableau de données
# TODO : extraire les données des pdf des étudiants
data = np.random.rand(100, 66)

# Extraire les réponses et les notes
responses = data[:, :-1]
grades = data[:, -1]

# Former un modèle de régression logistique pour chaque question
models = []
for i in range(13):
    model = LogisticRegression()
    model.fit(responses[:, i*5:(i+1)*5], grades)
    models.append(model)

# Utiliser les modèles pour prédire les réponses correctes
correct_answers = []
for model in models:
    # On suppose que la réponse avec la plus grande probabilité est la correcte
    probabilities = model.predict_proba(np.eye(5))
    correct_answer = np.argmax(probabilities, axis=0)
    correct_answers.append(correct_answer)

print(correct_answers)
