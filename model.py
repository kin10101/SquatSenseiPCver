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

# Filter the data based on the 'Phase' column
df_top_phase = df[df['Phase'] == 'top']
df_not_bottom_phase = df[df['Phase'] != 'bottom']
df_not_top_phase = df[df['Phase'] != 'top']

# Separate features for each target variable
x_stance = df_top_phase[['X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24', 'X31', 'Y31', 'Z31', 'X32', 'Y32', 'Z32']]
x_phase = df[['X11', 'Y11', 'Z11', 'X12', 'Y12', 'Z12', 'X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24',
              'X25', 'Y25', 'Z25', 'X26', 'Y26', 'Z26', 'X27', 'Y27', 'Z27', 'X28', 'Y28', 'Z28']]
x_head = df_not_bottom_phase[['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'X4', 'Y4', 'Z4', 'X5', 'Y5',
                              'Z5', 'X11', 'Y11', 'Z11', 'X12', 'Y12', 'Z12']]
x_chest = df_not_top_phase[['X11', 'Y11', 'Z11', 'X12', 'Y12', 'Z12', 'X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24']]
x_heel = df_not_top_phase[['X27', 'Y27', 'Z27', 'X28', 'Y28', 'Z28', 'X29', 'Y29', 'Z29', 'X30', 'Y30', 'Z30',
                           'X31', 'Y31', 'Z31', 'X32', 'Y32', 'Z32']]
x_knee = df_not_top_phase[['X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24', 'X25', 'Y25', 'Z25', 'X26', 'Y26', 'Z26',
                           'X31', 'Y31', 'Z31', 'X32', 'Y32', 'Z32']]

# Separate targets
y_stance = df_top_phase['Stance']
y_phase = df['Phase']
y_head = df_not_bottom_phase['Head']
y_chest = df_not_top_phase['Chest']
y_heel = df_not_top_phase['Heel']
y_knee = df_not_top_phase['Knee']

# Split the data into training and testing sets for each target variable
X_train_stance, X_test_stance, y_train_stance, y_test_stance = train_test_split(x_stance, y_stance, test_size=0.4, random_state=42)
X_train_phase, X_test_phase, y_train_phase, y_test_phase = train_test_split(x_phase, y_phase, test_size=0.4, random_state=42)
X_train_head, X_test_head, y_train_head, y_test_head = train_test_split(x_head, y_head, test_size=0.4, random_state=42)
X_train_chest, X_test_chest, y_train_chest, y_test_chest = train_test_split(x_chest, y_chest, test_size=0.4, random_state=42)
X_train_heel, X_test_heel, y_train_heel, y_test_heel = train_test_split(x_heel, y_heel, test_size=0.4, random_state=42)
X_train_knee, X_test_knee, y_train_knee, y_test_knee = train_test_split(x_knee, y_knee, test_size=0.4, random_state=42)

trees = 50
# Initialize Random Forest Classifiers for each target
clf_stance = RandomForestClassifier(n_estimators=trees, random_state=42)
clf_phase = RandomForestClassifier(n_estimators=trees, random_state=42)
clf_head = RandomForestClassifier(n_estimators=trees, random_state=42)
clf_chest = RandomForestClassifier(n_estimators=trees, random_state=42)
clf_heel = RandomForestClassifier(n_estimators=trees, random_state=42)
clf_knee = RandomForestClassifier(n_estimators=trees, random_state=42)

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


from sklearn.model_selection import cross_val_score

# Number of cross-validation folds
cv_folds = 5

# Cross-validation for each classifier
stance_cv_scores = cross_val_score(clf_stance, x_stance, y_stance, cv=cv_folds)
phase_cv_scores = cross_val_score(clf_phase, x_phase, y_phase, cv=cv_folds)
head_cv_scores = cross_val_score(clf_head, x_head, y_head, cv=cv_folds)
chest_cv_scores = cross_val_score(clf_chest, x_chest, y_chest, cv=cv_folds)
heel_cv_scores = cross_val_score(clf_heel, x_heel, y_heel, cv=cv_folds)
knee_cv_scores = cross_val_score(clf_knee, x_knee, y_knee, cv=cv_folds)

# Print cross-validation scores for each classifier
print("Stance Cross-Validation Scores:", stance_cv_scores)
print("Average Stance CV Score:", stance_cv_scores.mean())

print("\nPhase Cross-Validation Scores:", phase_cv_scores)
print("Average Phase CV Score:", phase_cv_scores.mean())

print("\nHead Cross-Validation Scores:", head_cv_scores)
print("Average Head CV Score:", head_cv_scores.mean())

print("\nChest Cross-Validation Scores:", chest_cv_scores)
print("Average Chest CV Score:", chest_cv_scores.mean())

print("\nHeel Cross-Validation Scores:", heel_cv_scores)
print("Average Heel CV Score:", heel_cv_scores.mean())

print("\nKnee Cross-Validation Scores:", knee_cv_scores)
print("Average Knee CV Score:", knee_cv_scores.mean())

