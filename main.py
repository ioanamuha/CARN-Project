import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import logging

from dataset import AmbiStoryDataset
from model import BertLSTM

logging.set_verbosity_error()

BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 25
EMBED_DIM = 100
HIDDEN_DIM = 128
MODEL_NAME = 'bert-base-uncased'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        targets = batch['target'].to(DEVICE)

        optimizer.zero_grad()

        predictions = model(input_ids, attention_mask)

        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def generate_submission(model, val_loader, output_file):
    model.eval()
    results = []

    print(f"Generating predictions to {output_file}...")

    with torch.no_grad():
        for batch in val_loader:
            ids = batch['id']

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            preds = model(input_ids, attention_mask)
            preds_list = preds.cpu().tolist()

            if not isinstance(preds_list, list):
                preds_list = [preds_list]

            for i, sample_id in enumerate(ids):
                score = preds_list[i]
                score = max(1, min(5, score))
                results.append({
                    "id": sample_id,
                    "prediction": score
                })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with torch.no_grad():
        with open(output_file, 'w') as f:
            for entry in results:
                json.dump(entry, f)
                f.write('\n')
    print("Done.")


def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            targets = batch['target'].to(DEVICE)

            predictions = model(input_ids, attention_mask)

            preds_list = predictions.cpu().tolist()
            if batch_idx == 0:
                print(f"\n[DEBUG] Sample Predictions from Dev Set: {preds_list[:5]}")

            loss = criterion(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main():
    print(f"Running on device: {DEVICE}")

    print("--- 1. Data Loading Phase ---")
    print("Building vocabulary...")

    train_dataset = AmbiStoryDataset('data/train.json', model_name=MODEL_NAME)
    val_dataset = AmbiStoryDataset('data/dev.json', model_name=MODEL_NAME)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset loaded. Train samples: {len(train_dataset)}")

    print(f"Initializing BertLSTM ({MODEL_NAME})...")
    model = BertLSTM(model_name=MODEL_NAME, hidden_dim=HIDDEN_DIM).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print("\n--- Starting Training ---")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        val_loss = validate_model(model, val_loader, criterion)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if (epoch + 1) == EPOCHS:
            torch.save(model.state_dict(), 'final_lstm_model.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            print(f"   >>> New Best Model Saved (Val Loss: {val_loss:.4f})")

    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load('best_lstm_model.pth'))

    generate_submission(model, val_loader, 'predictions/lstm_predictions_dev_best_model.jsonl')

    print("\n--- Running Official Evaluation ---")
    os.system(
        "python scoring.py data/ref/solution.jsonl predictions/lstm_predictions_dev_best_model.jsonl output/scores_best.json")

    print("\nLoading final model for evaluation...")
    model.load_state_dict(torch.load('final_lstm_model.pth'))

    generate_submission(model, val_loader, 'predictions/lstm_predictions_dev_final_model.jsonl')

    print("\n--- Running Official Evaluation ---")
    os.system(
        "python scoring.py data/ref/solution.jsonl predictions/lstm_predictions_dev_final_model.jsonl output/scores_final.json")


if __name__ == '__main__':
    main()
