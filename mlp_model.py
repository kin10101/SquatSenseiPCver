import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import keras
from keras import layers
from keras.src.utils import to_categorical
import matplotlib.pyplot as plt
import pickle

# Define the file paths
kel_path = r'E:\Pycharm Projects\Squat Sensei\Datasets\kel.csv'
kin_path = r'E:\Pycharm Projects\Squat Sensei\Datasets\kin.csv'
etc_path = r'E:\Pycharm Projects\Squat Sensei\Datasets\etc.csv'

# Read the CSV files
df_kel = pd.read_csv(kel_path)
df_kin = pd.read_csv(kin_path)
df_etc = pd.read_csv(etc_path)

df = pd.concat([df_kel, df_etc], ignore_index=True)

#convert string to int
df['Stance'] = df['Stance'].map({'wide': 0, 'standard': 1, 'narrow': 2})
df['Phase'] = df['Phase'].map({'top': 0, 'middle': 1, 'bottom': 2})

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

# Separate targets and convert them to categorical
y_stance = to_categorical(df['Stance'])
y_phase = to_categorical(df['Phase'])
y_head = to_categorical(df['Head'])
y_chest = to_categorical(df['Chest'])
y_heel = to_categorical(df['Heel'])
y_knee = to_categorical(df['Knee'])


# Split the data into training and testing sets for each target variable
X_train_stance, X_test_stance, y_train_stance, y_test_stance = train_test_split(x_stance, y_stance, test_size=0.2, random_state=42)
X_train_phase, X_test_phase, y_train_phase, y_test_phase = train_test_split(x_phase, y_phase, test_size=0.2, random_state=42)
X_train_head, X_test_head, y_train_head, y_test_head = train_test_split(x_head, y_head, test_size=0.2, random_state=42)
X_train_chest, X_test_chest, y_train_chest, y_test_chest = train_test_split(x_chest, y_chest, test_size=0.2, random_state=42)
X_train_heel, X_test_heel, y_train_heel, y_test_heel = train_test_split(x_heel, y_heel, test_size=0.2, random_state=42)
X_train_knee, X_test_knee, y_train_knee, y_test_knee = train_test_split(x_knee, y_knee, test_size=0.2, random_state=42)

# Function to create a Keras MLP model
def create_mlp(input_dim, output_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to plot training history
def plot_training_history(history, model_name):
    # Plot accuracy
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy over Epochs')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Train each classifier with validation data and plot their history
histories = {}  # Dictionary to store the histories of each model for plotting

# Initialize and train each model with validation split
clf_stance = create_mlp(X_train_stance.shape[1], y_train_stance.shape[1])
histories['Stance'] = clf_stance.fit(X_train_stance, y_train_stance, epochs=30, batch_size=32, verbose=1,
                                     validation_split=0.2)

clf_phase = create_mlp(X_train_phase.shape[1], y_train_phase.shape[1])
histories['Phase'] = clf_phase.fit(X_train_phase, y_train_phase, epochs=30, batch_size=32, verbose=1,
                                   validation_split=0.2)

clf_head = create_mlp(X_train_head.shape[1], y_train_head.shape[1])
histories['Head'] = clf_head.fit(X_train_head, y_train_head, epochs=30, batch_size=32, verbose=1, validation_split=0.2)

clf_chest = create_mlp(X_train_chest.shape[1], y_train_chest.shape[1])
histories['Chest'] = clf_chest.fit(X_train_chest, y_train_chest, epochs=30, batch_size=32, verbose=1,
                                   validation_split=0.2)

clf_heel = create_mlp(X_train_heel.shape[1], y_train_heel.shape[1])
histories['Heel'] = clf_heel.fit(X_train_heel, y_train_heel, epochs=30, batch_size=32, verbose=1, validation_split=0.2)

clf_knee = create_mlp(X_train_knee.shape[1], y_train_knee.shape[1])
histories['Knee'] = clf_knee.fit(X_train_knee, y_train_knee, epochs=30, batch_size=32, verbose=1, validation_split=0.2)

# Plot training histories
for model_name, history in histories.items():
    plot_training_history(history, model_name)

# Evaluate each model
loss_stance, accuracy_stance = clf_stance.evaluate(X_test_stance, y_test_stance, verbose=0)
loss_phase, accuracy_phase = clf_phase.evaluate(X_test_phase, y_test_phase, verbose=0)
loss_head, accuracy_head = clf_head.evaluate(X_test_head, y_test_head, verbose=0)
loss_chest, accuracy_chest = clf_chest.evaluate(X_test_chest, y_test_chest, verbose=0)
loss_heel, accuracy_heel = clf_heel.evaluate(X_test_heel, y_test_heel, verbose=0)
loss_knee, accuracy_knee = clf_knee.evaluate(X_test_knee, y_test_knee, verbose=0)

print("Stance Model Accuracy:", accuracy_stance)
print("Phase Model Accuracy:", accuracy_phase)
print("Head Model Accuracy:", accuracy_head)
print("Chest Model Accuracy:", accuracy_chest)
print("Heel Model Accuracy:", accuracy_heel)
print("Knee Model Accuracy:", accuracy_knee)

# Save each model to a separate file in the native Keras format
clf_stance.save('Models/stance_model.keras')
clf_phase.save('Models/phase_model.keras')
clf_head.save('Models/head_model.keras')
clf_chest.save('Models/chest_model.keras')
clf_heel.save('Models/heel_model.keras')
clf_knee.save('Models/knee_model.keras')
print('done training')