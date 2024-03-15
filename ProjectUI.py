import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size1, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, num_classes)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        x = nn.functional.softmax(x, dim=1)  # Apply softmax
        return x
            
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

# @st.cache_data
# def split_scale_transform(df: pd.DataFrame, y: pd.Series, testSize):
#     X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=testSize, random_state=42, shuffle=True)
    
#     # Fits the scaler on the training features
#     scaler = MinMaxScaler()
#     scaler.fit(X_train)

#     # Transforms the training and testing features
#     X_train_scaled = scaler.transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     return X_train_scaled, X_test_scaled, y_train, y_test, X_train

@st.cache_data
def encode_labels(y_train, y_test):
    # Encodes labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return y_train_encoded, y_test_encoded, label_encoder

@st.cache_data
def convert_tensor(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded):
    # Converts to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

@st.cache_resource(show_spinner="Training model...")
def train_model(X_train, _X_train_tensor, _y_train_tensor, _label_encoder):
    with st.spinner(text="Training..."):

       # Hyperparameters
        input_size = X_train.shape[1]
        hidden_size1 = 512
        num_classes = len(_label_encoder.classes_)
        learning_rate = 0.002
        num_epochs = 200
        batch_size = 32

        # Initializes model, loss function, and optimizer
        model = Classifier(input_size, hidden_size1, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = TensorDataset(_X_train_tensor, _y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step() 

        st.success("Done")
    return model

#@st.cache_data
def prepare_data(df, scaler):
    filenames = df.pop('filename')
    y = df.pop('label').values
    df.drop('length', axis=1)

    X_scaled = scaler.transform(df)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    return filenames, y, X_scaled, X_tensor, y_tensor


st.header("Can you beat the computer?")

inputSelection = st.selectbox("Game Mode :video_game:", ["30 Second","3 second (Takes a while to train)"],)

if inputSelection == "30 Second":
    df = load_data("Data/features_30_sec.csv")
else:
    df = load_data("Data/features_3_sec.csv")

filenames = df.pop('filename')
df = df.drop('length', axis=1)
y = df.pop('label').values
df_game = df.copy()
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, shuffle=True)
    
# Fits the scaler on the training features
scaler = MinMaxScaler()
scaler.fit(X_train)

# Transforms the training and testing features
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#filenames, acutal_genres, X_scaled, X_tensor, y_tensor = prepare_data(df_game, scaler)

game_scaled = scaler.transform(df_game)
game_tensor = torch.tensor(game_scaled, dtype=torch.float32)


#X_train_scaled, X_test_scaled, y_train, y_test, X_train = split_scale_transform(df, y, 0.2)
y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_tensor(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded)

model = train_model(X_train, X_train_tensor, y_train_tensor, label_encoder)


trackAmt = df_game.shape[0]
randomIndex = np.random.randint(1, trackAmt)
selectedTrack = filenames[randomIndex]
selectedGenre = selectedTrack.split(".",1)[0]
if inputSelection != "30 Second":
    words = selectedTrack.split(".",2)
    selectedTrack = words[0] + '.' +words[1] + ".wav"


st.audio("Data/genres_original/"+selectedGenre+"/"+selectedTrack)

model.eval()
with torch.no_grad():
    outputs = model(game_tensor)
    probabiltys, predicted = torch.max(outputs, 1)

    predicted_decoded = label_encoder.inverse_transform(predicted)

st.header("Computer's guess down below!")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")
st.markdown("####")

col1, col2 = st.columns(2)
col1.header("Computers Guess")
col1.write(predicted_decoded[randomIndex])

col2.header("Correct Genre")
col2.write(y[randomIndex])

st.button("Try another", type="primary")

# # Evaluates the model
# with torch.no_grad():
#     outputs = model(X_test_tensor)
#     probabilities, predicted = torch.max(outputs, 1)
#     accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
#     #print(f'Accuracy: {accuracy:.4f}')

#     # Decode labels
#     y_test_decoded = label_encoder.inverse_transform(y_test_tensor)
#     predicted_decoded = label_encoder.inverse_transform(predicted)

#     # Generate classification report
#     st.markdown("Classification Report:")
#     st.write(classification_report(y_test_decoded, predicted_decoded,output_dict=True))

# index = 50
# st.markdown(f"y_test_decoded: {y_test_decoded[index]}, predicted_decoded: {predicted_decoded[index]}, filename: {filenames[index]}")

# #st.dataframe(df)

