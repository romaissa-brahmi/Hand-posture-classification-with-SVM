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

# Initialize the SVM classifier with One-vs-All strategy
svm_ova = SVC(decision_function_shape='ovr')
svm_ova.fit(X_train, y_train)

# Predict and evaluate the One-vs-One model
y_pred_ova = svm_ova.predict(X_test)
cm = confusion_matrix(y_test, y_pred_ova)
accuracy = float(round(accuracy_score(y_test, y_pred_ova) * 100, 2))

print("One-vs-One Accuracy:", accuracy_score(y_test, y_pred_ova))

posture_names = ['Open', 'Closed', 'Thumb', 'Pinch', 'Almost', 'Trash', 'Point']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=posture_names)
disp.plot(cmap=plt.cm.Greens)
plt.title(f'Confusion Matrix - OVA - Accuracy = {accuracy}%')
plt.show()