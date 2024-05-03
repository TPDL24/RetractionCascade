import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

# Load data from A.csv and B.csv
df_A = pd.read_csv('A.csv', sep=';')
df_B = pd.read_csv('B.csv', sep=';')

# Preprocess data
df_A['Sentence'] = df_A['Sentence'].apply(lambda x: str(x))
df_B['Sentence'] = df_B['Sentence'].apply(lambda x: str(x))

# Define custom PyTorch dataset
class CustomDataset(Dataset):
    def __init__(self, sentences, labels=None, tokenizer=None, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])

        # Tokenize sentence and encode it
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            return input_ids, attention_mask, label
        else:
            return input_ids, attention_mask

# Split data into train and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    df_A['Sentence'], df_A['Label'], test_size=0.2, random_state=42
)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Assuming binary classification

# Define hyperparameters
batch_size = 32
epochs = 3
learning_rate = 2e-5

# Create datasets and dataloaders
train_dataset = CustomDataset(train_sentences, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomDataset(val_sentences, val_labels, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Evaluation on validation set
model.eval()
val_losses = []
val_predictions = []
val_true_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        val_loss = criterion(outputs.logits, labels)
        val_losses.append(val_loss.item())

        predicted_labels = torch.argmax(outputs.logits, dim=1)
        val_predictions.extend(predicted_labels.cpu().numpy())
        val_true_labels.extend(labels.cpu().numpy())

# Calculate validation metrics
val_loss = sum(val_losses) / len(val_losses)
val_accuracy = sum(val_predictions == val_true_labels) / len(val_true_labels)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Predict labels for B.csv
test_dataset = CustomDataset(df_B['Sentence'], tokenizer=tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

predictions = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predicted_labels = torch.argmax(outputs.logits, dim=1)
        predictions.extend(predicted_labels.cpu().numpy())

# Add predictions to df_B
df_B['Predicted_Label'] = predictions

# Save the updated B.csv with predicted labels
df_B.to_csv('B_with_predictions.csv', index=False)
