import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tqdm

from utils import * 

def train(train_list, train_label_map, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = EmbDataset(train_list, args.data_path, train_label_map, cache_data=True, file_format=args.file_format)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Determine dimensions
    input_dim = args.input_dim
    num_classes = args.num_classes
    if not input_dim or not num_classes:
        sample_data, _, _ = dataset[0]
        input_dim = sample_data.shape[0]
        num_classes = len(set(train_label_map.values()))

    model = ClassificationModel(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    accumulation_steps = args.accumulation_steps
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (batch_x, batch_y, batch_weights) in enumerate(
            tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        ):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_weights = batch_weights.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            weighted_loss = (loss * batch_weights).mean() / accumulation_steps

            weighted_loss.backward()

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += weighted_loss.item() * accumulation_steps

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train a classification model on imputed embeddings.")
    parser.add_argument(
        '--train_test_meta', '--meta', dest='meta', type=str, required=True,
        help="Path to the train/test split metadata pickle file."
    )
    parser.add_argument(
        '--data_path', type=str, default='.', help="Directory containing data files."
    )
    parser.add_argument(
        '--file_format', type=str, default='csv', choices=['csv', 'parquet'],
        help="File format of input data files."
    )
    parser.add_argument(
        '--sample_column', type=str, default='SubID',
        help="Column name for sample IDs in metadata."
    )
    parser.add_argument(
        '--phenotype_column', type=str, default='c15x',
        help="Column name for phenotype labels in metadata."
    )
    parser.add_argument(
        '--epochs', type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        '--input_dim', type=int, default=None,
        help="Dimensionality of input features (auto-inferred if not set)."
    )
    parser.add_argument(
        '--num_classes', type=int, default=None,
        help="Number of output classes (auto-inferred if not set)."
    )
    parser.add_argument(
        '--lr', type=float, default=0.01, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        '--accumulation_steps', type=int, default=1,
        help="Number of steps to accumulate gradients before updating."
    )
    parser.add_argument(
        '--num_workers', type=int, default=2,
        help="Number of DataLoader worker threads."
    )
    args = parser.parse_args()

    train_list, train_label_map = get_file_lists(
        args.meta, sample_column=args.sample_column, phen_column=args.phenotype_column
    )
    train(train_list, train_label_map, args)


if __name__ == '__main__':
    main()
