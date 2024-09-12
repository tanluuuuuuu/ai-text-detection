import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from binoculars import Binoculars

# Initialize the Binoculars object
bino = Binoculars()

# Load your test data
df = pd.read_csv('test.csv')  # Replace with your actual test dataset path
texts = df['text']
true_labels = df['generated']

# Initialize an empty list to store binoculars_scores
binoculars_scores_list = []

# Update the classify_text function to append scores to the list
def classify_text(text):
    prediction, binoculars_scores = bino.predict(text)
    binoculars_scores_list.append(binoculars_scores)  # Save the scores
    return 1 if 'AI-Generated' in prediction else 0  # Adjust this logic based on your exact output from bino.predict()

# Predict using Binoculars and collect scores
predictions = texts.apply(classify_text)

# Generate classification report
report = classification_report(true_labels, predictions, target_names=['Human', 'AI'])
print(report)
cm = confusion_matrix(true_labels, predictions)
print('Confusion matrix: \n', cm)

# Save the binoculars_scores to a txt file
with open('binoculars_scores.txt', 'w') as file:
    for score in binoculars_scores_list:
        file.write(f"{score}\n")