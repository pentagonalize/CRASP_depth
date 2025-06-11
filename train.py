import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.nn as nn
import json
import argparse
import os

class TransformerCharPredictor(nn.Module):
    def __init__(self, vocab_size, dim=5, heads=1, layers=3, k=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)

        # Final Linear Layer to project to k output symbols
        self.output = nn.Linear(dim, k)  # Project to k symbols (size of k)

    def generate_future_mask(self, size):
        # Causal mask: prevent attending to future tokens
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward(self, x):
        # Ensure x has a batch dimension
        if x.dim() == 1:  # If x is 1D (seq_len), add a batch dimension
            x = x.unsqueeze(0)  # (1, seq_len)

        # x: (batch, seq_len)
        x_embed = self.embedding(x)  # (batch, seq_len, dim)
        tgt_mask = self.generate_future_mask(x.size(1)).to(x.device)
        x_decoded = self.decoder(tgt=x_embed, memory=torch.zeros_like(x_embed), tgt_mask=tgt_mask)
        # Project output to k dimensions (symbols)
        logits = self.output(x_decoded)  # (batch, seq_len, k)
        probs = torch.sigmoid(logits)    # Probabilities for each of the k symbols (sigmoid activation)

        # Return both probabilities and the decoded output (to feed into the next step)
        return probs, x_decoded


def train(model, optimizer, device, source, target, k, epochs=10):
    model.train()
    criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_elements = 0
        data = tqdm(zip(source, target), total=len(source))
        print(f"Epoch {epoch + 1}/{epochs}")
        for src, tgt in data:
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()
            output, _ = model(src)

            # Reshape output and target for loss calculation
            output = output.view(-1, k)  # (batch * seq_len, k)
            tgt = tgt.view(-1)           # (batch * seq_len)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            # Take argmax of the output
            output = torch.argmax(output, dim=-1).squeeze(0)
            # Correct only if the output matches the target exactly
            correct = torch.equal(output, tgt)
            if correct:
                total_correct += 1
            total_elements += 1


            total_loss += loss.item()

        # Calculate total accuracy
        accuracy = total_correct / total_elements * 100

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(source):.4f}, Accuracy: {accuracy:.2f}%")

def evaluate(model, source, target, device, set_name=""):
    model.eval()
    total_correct = 0
    total_elements = 0

    with torch.no_grad():
        for src, tgt in tqdm(zip(source, target), total=len(source), desc=f"Evaluating on {set_name}"):
            src = src.to(device)
            tgt = tgt.to(device)

            output, _ = model(src)
            # Take argmax of the output
            output = torch.argmax(output, dim=-1).squeeze(0)

            correct = torch.equal(output, tgt)
            if correct:
                total_correct += 1
            total_elements += 1

    # Calculate total accuracy
    total_accuracy = total_correct / total_elements * 100
    print(f"Accuracy on {set_name}: {total_accuracy:.2f}%")
    return total_accuracy

def load_and_preprocess_data(input_path, file_name, bos_token):
    """Load and preprocess data from source and target files with BOS token.

    Args:
        input_path: Base path to data files
        file_name: Name of the data set (e.g., "50_to_100")
        bos_token: Token to use as beginning of sequence marker

    Returns:
        Tuple of (source, target) lists of tensors
    """
    src_path = os.path.join(input_path, f"L{k}", f"{file_name}_src.txt")
    tgt_path = os.path.join(input_path, f"L{k}", f"{file_name}_tgt.txt")

    with open(src_path, "r") as f:
        source = [line.strip() for line in f.readlines()]
    with open(tgt_path, "r") as f:
        target = [line.strip() for line in f.readlines()]

    # Convert source and target to tensors
    # Map 'a' to 0 and 'b' to 1
    source = [[0 if char == 'a' else 1 for char in line] for line in source]
    target = [[0 if char == '0' else 1 for char in line] for line in target]

    # Prepend BOS token to each source sequence
    source = [[bos_token] + line for line in source]
    # Prepend a target token (using 0) to maintain the same length as source
    target = [[0] + line for line in target]

    source = [torch.tensor(line) for line in source]
    target = [torch.tensor(line) for line in target]

    return source, target

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Set default values if not specified in the config file
    config.setdefault('heads', [1])
    config.setdefault('dims', [16, 64, 256])
    config.setdefault('lrs', [0.00001, 0.000001])
    config.setdefault('epochs', 10)
    config.setdefault('layers', 1)
    config.setdefault('input_path', 'data')
    config.setdefault('output_path', 'models')
    config.setdefault('train_split', 0.8)
    config.setdefault('bos_token', 2)
    config.setdefault('train_test_sets', {"L11": {"train_set": "50_to_100", "test_sets": ["50_to_100", "100_to_150", "150_to_200", "200_to_250"]}})

    return config

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train transformer models with configuration from JSON')
    parser.add_argument('--config', type=str, default='config.json', help='Path to JSON configuration file')
    args = parser.parse_args()

    # Load configuration from JSON file
    config = load_config(args.config)

    # Extract general configuration parameters
    heads = config['heads']
    dims = config['dims']
    lrs = config['lrs']
    epochs = config['epochs']
    layers = config['layers']
    input_path = config['input_path']
    output_path = config['output_path']
    train_split = config['train_split']
    bos_token = config['bos_token']

    vocab_size = 3  # Updated vocab size to include BOS token
    output_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducibility
    torch.manual_seed(42)

    for k_str, sets in config['train_test_sets'].items():
        k = int(k_str.replace("L", ""))
        train_set_name = sets['train_set']
        test_set_names = sets['test_sets']
        file_name = f"L{k}"
        print(f"Training with k = {k} and layers = {layers}")

        # Load training data
        source_train_full, target_train_full = load_and_preprocess_data(input_path, train_set_name, bos_token)

        # Split the training data into training and validation sets
        split_point = int(len(source_train_full) * train_split)
        source_train = source_train_full[:split_point]
        target_train = target_train_full[:split_point]
        source_eval = source_train_full[split_point:]
        target_eval = target_train_full[split_point:]

        # Create output directory if it doesn't exist
        os.makedirs(f"{output_path}/{file_name}", exist_ok=True)

        # clear the log file
        log_file_path = f"{output_path}/{file_name}/training_log.txt"
        with open(log_file_path, "w") as f:
            f.write("")

        log_file = open(log_file_path, "a")
        success = False
        final_heads = 0
        final_dim = 0
        final_lr = 0
        model = None
        best_model = None
        best_accuracy = 0

        # Write training configuration to log
        log_file.write(f"Training configuration:\n")
        log_file.write(f"Heads: {heads}\n")
        log_file.write(f"Dimensions: {dims}\n")
        log_file.write(f"Learning rates: {lrs}\n")
        log_file.write(f"Epochs: {epochs}\n")
        log_file.write(f"Layers: {layers}\n")
        log_file.write(f"Train set: {train_set_name}\n")
        log_file.write(f"Test sets: {test_set_names}\n\n")
        log_file.flush()

        for head in heads:
            if success:
                break
            for dim in dims:
                if success:
                    break
                for lr in lrs:
                    if success:
                        break
                    model = TransformerCharPredictor(vocab_size, dim, head, layers, output_size)
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr)
                    model_name = f"transformer_model_dim_{dim}_heads_{head}_layers_{layers}.pth"

                    print("Testing with dim:", dim, "heads:", head, "lr:", lr)
                    log_file.write(f"Testing with dim: {dim}, heads: {head}, lr: {lr}\n")
                    log_file.flush()

                    model.train()
                    criterion = CrossEntropyLoss()

                    for epoch in range(epochs):
                        total_loss = 0
                        total_correct = 0
                        total_elements = 0
                        data = tqdm(zip(source_train, target_train), total=len(source_train))
                        print(f"Epoch {epoch + 1}/{epochs}")
                        for src, tgt in data:
                            src = src.to(device)
                            tgt = tgt.to(device)

                            optimizer.zero_grad()
                            output, _ = model(src)

                            # Reshape output and target for loss calculation
                            output = output.view(-1, output_size)  # (batch * seq_len, k)
                            tgt = tgt.view(-1)                     # (batch * seq_len)

                            loss = criterion(output, tgt)
                            loss.backward()
                            optimizer.step()

                            # Take argmax of the output
                            output = torch.argmax(output, dim=-1).squeeze(0)
                            # Correct only if the output matches the target exactly
                            correct = torch.equal(output, tgt)
                            if correct:
                                total_correct += 1
                            total_elements += 1

                            total_loss += loss.item()

                        # Calculate total accuracy on the training set
                        train_accuracy = total_correct / total_elements * 100
                        print(f"Epoch {epoch + 1} Loss: {total_loss / len(source_train):.4f}, Training Accuracy: {train_accuracy:.2f}%")

                        # Write to log file
                        log_file.write(f"Epoch {epoch + 1}, Loss: {total_loss / len(source_train):.4f}, Training Accuracy: {train_accuracy:.2f}%\n")
                        log_file.flush()

                        # Evaluate the model on the validation set
                        accuracy = evaluate(model, source_eval, target_eval, device, set_name="Validation Set")

                        if accuracy == 100:
                            print(f"Model {model_name} reached 100% validation accuracy")
                            success = True
                            final_heads = head
                            final_dim = dim
                            final_lr = lr
                            break
                        elif accuracy >= best_accuracy:
                            best_accuracy = accuracy
                            final_heads = head
                            final_dim = dim
                            final_lr = lr
                            best_model = model.state_dict().copy()
                if success:
                    break
            if success:
                break

        # Save the best model
        if success:
            torch.save(model.state_dict(), f"{output_path}/{file_name}/{model_name}")
            current_model = model

            # Write model details to log file
            log_file.write(f"Model {model_name} reached 100% validation accuracy\n")
            log_file.write(f"heads: {final_heads}, dim: {final_dim}, lr: {final_lr}\n\n")

            # Evaluate the model on all test files
            log_file.write("Evaluation results:\n")
            for test_set in test_set_names:
                test_source, test_target = load_and_preprocess_data(input_path, test_set, bos_token)
                accuracy = evaluate(current_model, test_source, test_target, device)
                log_file.write(f"Accuracy on {test_set}: {accuracy:.2f}%\n")

            log_file.close()

        else:
            # Record failure in log file
            log_file.write(f"All models failed to reach 100% validation accuracy. Saving best model on validation set.\n")
            model_name = f"transformer_model_dim_{final_dim}_heads_{final_heads}_layers_{layers}_best_on_validation.pth"

            # Use the best model for evaluation
            model = TransformerCharPredictor(vocab_size, final_dim, final_heads, layers, output_size)
            model.load_state_dict(best_model)
            model.to(device)
            current_model = model

            torch.save(best_model, f"{output_path}/{file_name}/{model_name}")

            # Write model details to log file
            log_file.write(f"Best model - heads: {final_heads}, dim: {final_dim}, lr: {final_lr}\n\n")

            # Evaluate the model on all test files
            log_file.write("Evaluation results:\n")
            for test_set in test_set_names:
                test_source, test_target = load_and_preprocess_data(input_path, test_set, bos_token)
                accuracy = evaluate(current_model, test_source, test_target, device)
                log_file.write(f"Accuracy on {test_set}: {accuracy:.2f}%\n")

            log_file.close()
