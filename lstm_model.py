import pandas as pd
import numpy as np
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras import Sequential
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


def reshape_to_sequences(data, seq_length=10):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequence = data[i:i + seq_length]
        sequences.append(sequence)
    return np.array(sequences)


x_stance = reshape_to_sequences(
    df[['X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24', 'X31', 'Y31', 'Z31', 'X32', 'Y32', 'Z32']])

x_phase = reshape_to_sequences(
    df[['X11', 'Y11', 'Z11', 'X12', 'Y12', 'Z12', 'X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24',
              'X25', 'Y25', 'Z25', 'X26', 'Y26', 'Z26', 'X27', 'Y27', 'Z27', 'X28', 'Y28', 'Z28']])

x_head = reshape_to_sequences(df[['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'X4', 'Y4', 'Z4', 'X5', 'Y5',
             'Z5', 'X11', 'Y11', 'Z11', 'X12', 'Y12', 'Z12']])

x_chest = reshape_to_sequences(df[['X11', 'Y11', 'Z11', 'X12', 'Y12', 'Z12', 'X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24']])

x_heel = reshape_to_sequences(df[['X27', 'Y27', 'Z27', 'X28', 'Y28', 'Z28', 'X29', 'Y29', 'Z29', 'X30', 'Y30', 'Z30',
             'X31', 'Y31', 'Z31', 'X32', 'Y32', 'Z32']])

x_knee = reshape_to_sequences(df[['X23', 'Y23', 'Z23', 'X24', 'Y24', 'Z24', 'X25', 'Y25', 'Z25', 'X26', 'Y26', 'Z26',
             'X31', 'Y31', 'Z31', 'X32', 'Y32', 'Z32']])

y_stance = to_categorical(df['Stance'])
y_phase = to_categorical(df['Phase'])
y_head = to_categorical(df['Head'])
y_chest = to_categorical(df['Chest'])
y_heel = to_categorical(df['Heel'])
y_knee = to_categorical(df['Knee'])


# Split data into training and testing sets
X_train_stance, X_test_stance, y_train_stance, y_test_stance = train_test_split(x_stance, y_stance, test_size=0.2, random_state=42)
X_train_phase, X_test_phase, y_train_phase, y_test_phase = train_test_split(x_phase, y_phase, test_size=0.2, random_state=42)
X_train_head, X_test_head, y_train_head, y_test_head = train_test_split(x_head, y_head, test_size=0.2, random_state=42)
X_train_chest, X_test_chest, y_train_chest, y_test_chest = train_test_split(x_chest, y_chest, test_size=0.2, random_state=42)
X_train_heel, X_test_heel, y_train_heel, y_test_heel = train_test_split(x_heel, y_heel, test_size=0.2, random_state=42)
X_train_knee, X_test_knee, y_train_knee, y_test_knee = train_test_split(x_knee, y_knee, test_size=0.2, random_state=42)

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
# Train and evaluate stance model
model_stance = build_lstm_model(X_train_stance.shape[1:], y_train_stance.shape[1])
model_stance.fit(X_train_stance, y_train_stance, epochs=10, batch_size=32, validation_split=0.2)



# Predict and evaluate stance model
y_pred_stance = model_stance.predict(X_test_stance)
y_pred_stance = np.argmax(y_pred_stance, axis=1)
y_test_stance = np.argmax(y_test_stance, axis=1)

print("Stance Classification Report:")
print(classification_report(y_test_stance, y_pred_stance))
print("Accuracy:", accuracy_score(y_test_stance, y_pred_stance))

# Save the model
model_stance.save('Models/stance_lstm_model.h5')