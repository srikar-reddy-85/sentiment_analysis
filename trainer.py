# import pandas as pd
# import numpy as np
# import seaborn as sns
# # import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import RandomForestClassifier
# import joblib
# import re
#
# # Function to clean text
# def clean_text(text):
#     # Remove special characters and digits
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     # Convert to lowercase
#     text = text.lower()
#     return text
#
# # Load the dataset
# df = pd.read_csv("./processed_data.csv")
#
# # Display the first few rows and value counts
# print(df.head())
# print("\nAct value counts:")
# print(df['act'].value_counts())
# print("\nEmotion value counts:")
# print(df['emotion'].value_counts())
#
#
#
# # Clean the text
# df['clean_text'] = df['text'].apply(clean_text)
#
# # Prepare the features and targets
# X = df['clean_text']
# y = df[['act', 'emotion']]
#
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # Create and train the pipeline
# pipe = Pipeline([
#     ('cv', CountVectorizer()),
#     ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
# ])
#
# pipe.fit(X_train, y_train)
#
# # Evaluate the model
# train_score = pipe.score(X_train, y_train)
# test_score = pipe.score(X_test, y_test)
#
# print(f"Training score: {train_score:.4f}")
# print(f"Testing score: {test_score:.4f}")
#
# # Save the model
# joblib.dump(pipe, "text_emotion_act.pkl")
#
# # Function to predict act and emotion
# def predict_act_emotion(text):
#     # Clean the input text
#     clean_text_input = clean_text(text)
#
#     # Make prediction
#     prediction = pipe.predict([clean_text_input])[0]
#
#     # Map predictions to labels
#     act_labels = {1: "Statement or declaration", 2: "Question", 3: "Suggestion or proposal", 4: "Agreement or acceptance"}
#     emotion_labels = {0: "Neutral", 1: "Minimal discomfort", 2: "Discontent, frustration", 
#                       3: "Fear or nervousness", 4: "Positive (happiness, excitement, etc.)", 
#                       5: "Anger", 6: "Surprise or disbelief"}
#
#     predicted_act = act_labels[prediction[0]]
#     predicted_emotion = emotion_labels[prediction[1]]
#
#     return predicted_act, predicted_emotion
#
# # Example usage
# example_text = "Say, Jim, how about going for a few beers after dinner?"
# predicted_act, predicted_emotion = predict_act_emotion(example_text)
# print(f"Text: {example_text}")
# print(f"Predicted Act: {predicted_act}")
# print(f"Predicted Emotion: {predicted_emotion}")
#==========================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import joblib
import re
import warnings
warnings.filterwarnings("ignore")

# Function to clean text
def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Load the dataset
df = pd.read_csv("./processed_data.csv")

# Display the first few rows and value counts
print(df.head())
print("\nAct value counts:")
print(df['act'].value_counts())
print("\nEmotion value counts:")
print(df['emotion'].value_counts())

# Clean the text
df['clean_text'] = df['text'].apply(clean_text)

# Prepare the features and targets
X = df['clean_text']
y_act = df['act'] - 1  # Subtract 1 to make classes 0-indexed
y_emotion = df['emotion']

# Split the data
X_train, X_test, y_act_train, y_act_test, y_emotion_train, y_emotion_test = train_test_split(
    X, y_act, y_emotion, test_size=0.3, random_state=42
)

# Load RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base')

# Custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, act_labels, emotion_labels, tokenizer, max_len=128):
        self.texts = texts
        self.act_labels = act_labels
        self.emotion_labels = emotion_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        act_label = self.act_labels[item]
        emotion_label = self.emotion_labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'act_labels': torch.tensor(act_label, dtype=torch.long),
            'emotion_labels': torch.tensor(emotion_label, dtype=torch.long)
        }

# Create datasets
train_dataset = TextDataset(X_train.values, y_act_train.values, y_emotion_train.values, tokenizer)
test_dataset = TextDataset(X_test.values, y_act_test.values, y_emotion_test.values, tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Custom model
class RobertaClassifier(nn.Module):
    def __init__(self, n_act_classes=4, n_emotion_classes=7):
        super(RobertaClassifier, self).__init__()
        self.roberta = roberta
        self.drop = nn.Dropout(p=0.3)
        self.act_out = nn.Linear(self.roberta.config.hidden_size, n_act_classes)
        self.emotion_out = nn.Linear(self.roberta.config.hidden_size, n_emotion_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        act_output = self.act_out(output)
        emotion_output = self.emotion_out(output)
        return act_output, emotion_output

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
act_criterion = nn.CrossEntropyLoss().to(device)
emotion_criterion = nn.CrossEntropyLoss().to(device)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        act_labels = batch['act_labels'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)

        optimizer.zero_grad()
        act_output, emotion_output = model(input_ids, attention_mask)
        act_loss = act_criterion(act_output, act_labels)
        emotion_loss = emotion_criterion(emotion_output, emotion_labels)
        loss = act_loss + emotion_loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
act_preds, emotion_preds = [], []
act_labels, emotion_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        act_output, emotion_output = model(input_ids, attention_mask)
        
        act_preds.extend(torch.argmax(act_output, dim=1).cpu().numpy())
        emotion_preds.extend(torch.argmax(emotion_output, dim=1).cpu().numpy())
        act_labels.extend(batch['act_labels'].cpu().numpy())
        emotion_labels.extend(batch['emotion_labels'].cpu().numpy())

print("\nAct Classification Report:")
print(classification_report(act_labels, act_preds))
print("\nEmotion Classification Report:")
print(classification_report(emotion_labels, emotion_preds))

# Save the model
torch.save(model.state_dict(), "roberta_emotion_act_model.pth")

# Function to predict act and emotion
def predict_act_emotion(text):
    model.eval()
    clean_text_input = clean_text(text)
    encoding = tokenizer.encode_plus(
        clean_text_input,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        act_output, emotion_output = model(input_ids, attention_mask)
    
    act_pred = torch.argmax(act_output, dim=1).item()
    emotion_pred = torch.argmax(emotion_output, dim=1).item()
    
    act_labels = {0: "Statement or declaration", 1: "Question", 2: "Suggestion or proposal", 3: "Agreement or acceptance"}
    emotion_labels = {0: "Neutral", 1: "Minimal discomfort", 2: "Discontent, frustration", 
                      3: "Fear or nervousness", 4: "Positive (happiness, excitement, etc.)", 
                      5: "Anger", 6: "Surprise or disbelief"}
    
    return act_labels[act_pred], emotion_labels[emotion_pred]

# Example usage
example_text = "Say, Jim, how about going for a few beers after dinner?"
predicted_act, predicted_emotion = predict_act_emotion(example_text)
print(f"Text: {example_text}")
print(f"Predicted Act: {predicted_act}")
print(f"Predicted Emotion: {predicted_emotion}")

