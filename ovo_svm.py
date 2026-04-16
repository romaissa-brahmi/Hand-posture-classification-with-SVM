import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv('data/hand_data.csv')
X = df.drop(['image_index', 'label'], axis=1)
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Initialize the SVM classifier with One-vs-One strategy
svm_ovo = SVC(decision_function_shape='ovo')
svm_ovo.fit(X_train, y_train)

# Predict and evaluate the One-vs-One model
y_pred_ovo = svm_ovo.predict(X_test)
cm = confusion_matrix(y_test, y_pred_ovo)
accuracy = float(round(accuracy_score(y_test, y_pred_ovo) * 100, 2))

print("One-vs-One Accuracy:", accuracy_score(y_test, y_pred_ovo))

posture_names = ['Open', 'Closed', 'Thumb', 'Pinch', 'Almost', 'Trash', 'Point']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=posture_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - OVO - Accuracy = {accuracy}%')
plt.show()