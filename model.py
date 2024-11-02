import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Define the file paths
kel_path = r'E:\Pycharm Projects\Squat Sensei\Datasets\kel.csv'
kin_path = r'E:\Pycharm Projects\Squat Sensei\Datasets\kin.csv'
etc_path = r'E:\Pycharm Projects\Squat Sensei\Datasets\etc.csv'

# Read the CSV files
df_kel = pd.read_csv(kel_path)
df_kin = pd.read_csv(kin_path)
df_etc = pd.read_csv(etc_path)

df = pd.concat([df_kel, df_kin, df_etc], ignore_index=True)

# Separate features for each target variable
x_stance = df[['X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24', 'X31', 'Y31', 'Z31', 'X32', 'Y32', 'Z32']]
x_phase = df[['X11', 'Y11', 'Z11', 'X12', 'Y12', 'Z12', 'X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24',
              'X25', 'Y25', 'Z25', 'X26', 'Y26', 'Z26', 'X27', 'Y27', 'Z27', 'X28', 'Y28', 'Z28']]
x_head = df[['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'X4', 'Y4', 'Z4', 'X5', 'Y5',
             'Z5', 'X11', 'Y11', 'Z11', 'X12', 'Y12', 'Z12']]
x_chest = df[['X11', 'Y11', 'Z11', 'X12', 'Y12', 'Z12', 'X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24']]
x_heel = df[['X27', 'Y27', 'Z27', 'X28', 'Y28', 'Z28', 'X29', 'Y29', 'Z29', 'X30', 'Y30', 'Z30',
             'X31', 'Y31', 'Z31', 'X32', 'Y32', 'Z32']]
x_knee = df[['X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24', 'X25', 'Y25', 'Z25', 'X26', 'Y26', 'Z26',
             'X31', 'Y31', 'Z31', 'X32', 'Y32', 'Z32']]

# Separate targets
y_stance = df['Stance']
y_phase = df['Phase']
y_head = df['Head']
y_chest = df['Chest']
y_heel = df['Heel']
y_knee = df['Knee']

# Split the data into training and testing sets for each target variable
X_train_stance, X_test_stance, y_train_stance, y_test_stance = train_test_split(x_stance, y_stance, test_size=0.2, random_state=42)
X_train_phase, X_test_phase, y_train_phase, y_test_phase = train_test_split(x_phase, y_phase, test_size=0.2, random_state=42)
X_train_head, X_test_head, y_train_head, y_test_head = train_test_split(x_head, y_head, test_size=0.2, random_state=42)
X_train_chest, X_test_chest, y_train_chest, y_test_chest = train_test_split(x_chest, y_chest, test_size=0.2, random_state=42)
X_train_heel, X_test_heel, y_train_heel, y_test_heel = train_test_split(x_heel, y_heel, test_size=0.2, random_state=42)
X_train_knee, X_test_knee, y_train_knee, y_test_knee = train_test_split(x_knee, y_knee, test_size=0.2, random_state=42)

# Initialize Random Forest Classifiers for each target
clf_stance = RandomForestClassifier(n_estimators=100, random_state=42)
clf_phase = RandomForestClassifier(n_estimators=100, random_state=42)
clf_head = RandomForestClassifier(n_estimators=100, random_state=42)
clf_chest = RandomForestClassifier(n_estimators=100, random_state=42)
clf_heel = RandomForestClassifier(n_estimators=100, random_state=42)
clf_knee = RandomForestClassifier(n_estimators=100, random_state=42)

# Train each classifier
clf_stance.fit(X_train_stance, y_train_stance)
clf_phase.fit(X_train_phase, y_train_phase)
clf_head.fit(X_train_head, y_train_head)
clf_chest.fit(X_train_chest, y_train_chest)
clf_heel.fit(X_train_heel, y_train_heel)
clf_knee.fit(X_train_knee, y_train_knee)

# Make predictions on the test set
y_pred_stance = clf_stance.predict(X_test_stance)
y_pred_phase = clf_phase.predict(X_test_phase)
y_pred_head = clf_head.predict(X_test_head)
y_pred_chest = clf_chest.predict(X_test_chest)
y_pred_heel = clf_heel.predict(X_test_heel)
y_pred_knee = clf_knee.predict(X_test_knee)

# Evaluate the models
print("Stance Classification Report:")
print(classification_report(y_test_stance, y_pred_stance))
print("Accuracy:", accuracy_score(y_test_stance, y_pred_stance))

print("\nPhase Classification Report:")
print(classification_report(y_test_phase, y_pred_phase))
print("Accuracy:", accuracy_score(y_test_phase, y_pred_phase))

print("\nHead Classification Report:")
print(classification_report(y_test_head, y_pred_head))
print("Accuracy:", accuracy_score(y_test_head, y_pred_head))

print("\nChest Classification Report:")
print(classification_report(y_test_chest, y_pred_chest))
print("Accuracy:", accuracy_score(y_test_chest, y_pred_chest))

print("\nHeel Classification Report:")
print(classification_report(y_test_heel, y_pred_heel))
print("Accuracy:", accuracy_score(y_test_heel, y_pred_heel))

print("\nKnee Classification Report:")
print(classification_report(y_test_knee, y_pred_knee))
print("Accuracy:", accuracy_score(y_test_knee, y_pred_knee))

# Save each model to a separate file
with open('Models/stance_model.pkl', 'wb') as file:
    pickle.dump(clf_stance, file)
with open('Models/phase_model.pkl', 'wb') as file:
    pickle.dump(clf_phase, file)
with open('Models/head_model.pkl', 'wb') as file:
    pickle.dump(clf_head, file)
with open('Models/chest_model.pkl', 'wb') as file:
    pickle.dump(clf_chest, file)
with open('Models/heel_model.pkl', 'wb') as file:
    pickle.dump(clf_heel, file)
with open('Models/knee_model.pkl', 'wb') as file:
    pickle.dump(clf_knee, file)
