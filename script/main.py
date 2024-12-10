from dataset import load_and_prepare_dataset
from model import GeneClassificationTransformer
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import os, argparse, sys
from torch.optim import Adam

def train_model(train_data, tokenizer, seq_len, d_model, epoch, model_save):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    vocab_size = len(tokenizer)

    data_path = train_data

    train_dataloader, val_dataloader = load_and_prepare_dataset(data_path)

    seq_len = int(seq_len)
    d_model = int(d_model)
    num_classes = 2
    model = GeneClassificationTransformer(vocab_size, seq_len, d_model=d_model, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-5)

    epochs = int(epoch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for batch in val_dataloader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    if model_save is None:
        torch.save(model.state_dict(), os.path.join("output","trained_model.pth"))
    else:
        torch.save(model.state_dict(), os.path.join("output", model_save))

def eval_model(model, tokenizer, sequence=None, sequence_file=None, seq_len=None, device=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sequence_dict = {}
    if sequence_file:
        # Parse the input file into a dictionary
        with open(sequence_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    print(f"Skipping malformed line: {line}")
                    continue
                name, seq = parts
                sequence_dict[name] = seq
    elif sequence:
        sequence_dict["Sequence1"] = sequence
    else:
        raise ValueError("Either `sequence` or `sequence_file` must be provided.")

    names = list(sequence_dict.keys())
    sequences = list(sequence_dict.values())

    # Tokenize the sequences
    inputs = tokenizer(
        sequences,
        truncation=True,
        padding="max_length",
        max_length=int(seq_len),
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=-1).cpu().numpy()

    # Write predictions to an output file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predictions.txt")

    gene_count = 0
    non_gene_count = 0

    with open(output_file, "w") as out:
        out.write("Name\tPredicted_Label\tSequence\n")
        for name, seq, pred in zip(names, sequences, predictions):
            label = "gene" if pred == 1 else "non-gene"
            if pred == 1:
                gene_count += 1
            else:
                non_gene_count += 1
            out.write(f"{name}\t{label}\t{seq}\n")

    # Print statistics
    total_sequences = len(sequences)
    print(f"Predictions saved to: {output_file}")
    print("\nStatistics:")
    print(f"Total Sequences: {total_sequences}")
    print(f"Gene Predictions: {gene_count} ({(gene_count / total_sequences) * 100:.2f}%)")
    print(f"Non-Gene Predictions: {non_gene_count} ({(non_gene_count / total_sequences) * 100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Main code for either training or evaluating the model for gene prediction.")

    parser.add_argument("--train", help="Train the model.", action="store_true")
    parser.add_argument("--eval", help="Evaluate the model.", action="store_true")

    parser.add_argument("--train_data", help="Path to training data. Data will be split Training and Validation.", required="--train" in sys.argv)
    parser.add_argument("--tokenizer", help="Link to a pretrained model that has tokenizer.", required="--train" in sys.argv)
    parser.add_argument("--seq_len", help="Size of the vocabulary", required="--train" or "--eval" in sys.argv)
    parser.add_argument("--d_model", help="Size of the model", required="--train" in sys.argv)
    parser.add_argument("--epoch", help="Value for batch training", required="--train" in sys.argv)
    parser.add_argument("--model_save", help="Name of the model file to be saved in")

    parser.add_argument("--model", help="Path to model file that has been trained", required="--eval" in sys.argv)
    parser.add_argument("--sequence", help="Path to model file that has been trained")
    parser.add_argument("--sequence_file", help="Path to model file that has been trained")
    parser.add_argument("--device", help="Can either be 'cuda' or 'cpu'.")

    args = parser.parse_args()

    if args.train:
        train_model(args.train_data, args.tokenizer, args.seq_len, args.d_model, args.epoch, args.model_save)

    if args.eval:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        vocab_size = len(tokenizer)
        d_model = 512
        seq_len = int(args.seq_len)
        model = GeneClassificationTransformer(vocab_size, seq_len, d_model=d_model, num_classes=2)
        
        model.load_state_dict(torch.load(args.model, weights_only=True))

        eval_model(model, tokenizer, sequence=args.sequence, sequence_file=args.sequence_file, seq_len=seq_len, device=args.device)

if __name__ == "__main__":
    main()