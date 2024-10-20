import torch
from transformers import RobertaTokenizer
import re
from transformers import RobertaModel
import warnings
warnings.filterwarnings("ignore")

# Load the tokenizer and model architecture
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Function to clean the input text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
    text = text.lower()  # Convert to lowercase
    return text

# Define the RobertaClassifier model (must match the architecture used during training)
class RobertaClassifier(torch.nn.Module):
    def __init__(self, n_act_classes=4, n_emotion_classes=7):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.drop = torch.nn.Dropout(p=0.3)
        self.act_out = torch.nn.Linear(self.roberta.config.hidden_size, n_act_classes)
        self.emotion_out = torch.nn.Linear(self.roberta.config.hidden_size, n_emotion_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.drop(pooled_output)
        act_output = self.act_out(output)
        emotion_output = self.emotion_out(output)
        return act_output, emotion_output

# Load the saved model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaClassifier().to(device)
model.load_state_dict(torch.load("./roberta_emotion_act_model.pth", map_location=device))
model.eval()

# Function to predict act and emotion
def predict_act_emotion(text):
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
    emotion_labels = {0: "Neutral", 1: "discomfort", 2: "Discontent, frustration", 
                      3: "Fear or nervousness", 4: "Positive (happiness, excitement, etc.)", 
                      5: "Anger", 6: "Surprise or disbelief"}

    return act_labels[act_pred], emotion_labels[emotion_pred]

# Main function to take user input and display the prediction
def main():
    while True:
        user_input = input("Enter a sentence (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        act, emotion = predict_act_emotion(user_input)
        print(f"\nPredicted Act: {act}")
        print(f"Predicted Emotion: {emotion}\n")

if __name__ == "__main__":
    main()

