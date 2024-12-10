import pandas as pd
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g")

def preprocess_sequence(sequence, max_len=512):
    tokenized = tokenizer(sequence, truncation=True, padding="max_length", max_length=max_len)
    return tokenized["input_ids"], tokenized["attention_mask"]

def load_and_prepare_dataset(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["chromosome", "gene_id", "label", "sequence"])

    input_ids = []
    attention_masks = []
    labels = []

    for _, row in df.iterrows():
        seq = row["sequence"]
        label = 1 if row["label"] == "gene" else 0
        token_ids, attention_mask = preprocess_sequence(seq)
        input_ids.append(token_ids)
        attention_masks.append(attention_mask)
        labels.append(label)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
