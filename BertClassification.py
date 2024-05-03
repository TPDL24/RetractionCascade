import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Predict the response for the test dataset



# Load the data from the CSV file
df = pd.read_csv('All_Seperate_citations_Labeled.csv', encoding='latin1')

# Preprocess the data (adjust as needed)
df['citation'] = df['citation'].apply(lambda x: str(x).lower() if isinstance(x, str) else '')

# Encode the 'Retracted' column to numerical labels
label_encoder = LabelEncoder()
df['Retracted'] = label_encoder.fit_transform(df['Retracted'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['citation'], df['Retracted'], test_size=0.2, random_state=42)

# Define BERT tokenizer and load pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Two classes: Retracted (1) or not (0)

# Tokenize the text data and create data loaders
max_seq_len = 100  # You may adjust this as needed
X_train_tokens = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=max_seq_len, return_tensors='pt')
X_test_tokens = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=max_seq_len, return_tensors='pt')

y_train_tensor = torch.tensor(y_train.tolist())
y_test_tensor = torch.tensor(y_test.tolist())

train_dataset = TensorDataset(X_train_tokens.input_ids, X_train_tokens.attention_mask, y_train_tensor)
test_dataset = TensorDataset(X_test_tokens.input_ids, X_test_tokens.attention_mask, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Set up optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)  # You may adjust the learning rate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
epochs = 5  # You may adjust the number of epochs
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(predicted_labels)
        true_labels.extend(labels.cpu().numpy())

print(classification_report(true_labels, predictions))
