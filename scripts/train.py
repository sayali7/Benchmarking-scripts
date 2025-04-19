import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score
import tqdm

from utils import * 


def train(train_list, train_label_map, val_list, val_label_map, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler=None
    # scaler = StandardScaler()
    # for donor in tqdm.tqdm(train_list, desc="Fitting scaler"):
    #     path = os.path.join(args.data_path, donor + f".{args.file_format}")
    #     df = pd.read_csv(path, index_col=0) if args.file_format == "csv" else pd.read_parquet(path)
    #     scaler.partial_fit(df.values)

    train_dataset = EmbDataset(train_list, args.data_path, train_label_map,
                               scaler=scaler, cache_data=True, file_format=args.file_format)
    val_dataset = EmbDataset(val_list, args.data_path, val_label_map,
                             scaler=scaler, cache_data=True, file_format=args.file_format)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    
    if args.input_dim:
        input_dim = args.input_dim
    else:
        sample_x, _, _ = train_dataset[0]
        input_dim = sample_x.shape[0]

    if args.num_classes:
        num_classes = args.num_classes
    else:
        num_classes = len(set(train_label_map.values()))

    hidden_dims = [int(h) for h in args.hidden_dims.split(",")] if args.hidden_dims else []

    label_counts = Counter(train_label_map.values())
    class_weights = [0] * num_classes
    for cls, count in label_counts.items():
        class_weights[cls] = 1.0 / count
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    if num_classes==2:
        num_classes=1
    model = MLP(input_dim, hidden_dims, num_classes).to(device)
    #criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none').to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    accumulation_steps = args.accumulation_steps
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        optimizer.zero_grad()

        for i, (bx, by, batch_weights) in enumerate(
            tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        ):
            bx = bx.to(device)
            by = by.to(device)
            batch_weights = batch_weights.to(device)
            #print (model)

            logits = model(bx)
            # labels_onehot = F.one_hot(by, num_classes).float()
            # logits = logits.squeeze(1)
            by = by.unsqueeze(1)
            #print (logits.shape, by.shape)
            loss = criterion(logits, by)

            loss = (loss).mean()

            if accumulation_steps>0:    
                train_loss /= accumulation_steps

            loss.backward()

            # Gradient accumulation
            if  accumulation_steps>0 and ((i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            if  accumulation_steps>0:
                 loss *= accumulation_steps

            train_loss += loss.item()

            #preds = torch.argmax(torch.sigmoid(logits), dim=1).cpu().tolist()
            preds = (logits > 0).long().cpu().tolist()
            train_preds.extend(preds)
            train_labels.extend(by.cpu().tolist())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = sum(p==t for p,t in zip(train_preds, train_labels)) / len(train_labels)
        train_bal_acc = balanced_accuracy_score(train_labels, train_preds)


        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for bx, by, batch_weights in tqdm.tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{args.epochs}"):
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                #labels_onehot = F.one_hot(by, num_classes).float()
                by = by.unsqueeze(1) 
                loss = criterion(logits, by)
                val_loss += loss.mean().item()
                #preds = torch.argmax(torch.sigmoid(logits), dim=1).cpu().tolist()
                preds = (logits > 0).long().cpu().tolist()
                val_preds.extend(preds)
                val_labels.extend(by.cpu().tolist())
        avg_val_loss = val_loss / len(val_loader)
        val_acc = sum(p==t for p,t in zip(val_preds, val_labels)) / len(val_labels)
        val_bal_acc = balanced_accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, Bal Acc: {train_bal_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, Bal Acc: {val_bal_acc:.4f}")
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train a classification model on cell embeddings.")
    parser.add_argument(
        '--meta', dest='meta', type=str,
        default="/media/sayalialatkar/T9/Sayali/FoundationModels/scBrainLLM/phenotype_classification_files/c15x_split_seed42.pkl", 
        help="Path to the train/test split metadata pickle file."
    )
    parser.add_argument(
        '--data_path', type=str, default='.', help="Directory containing data files."
    )
    parser.add_argument(
        '--file_format', type=str, default='csv', choices=['csv', 'parquet'],
        help="File format of input data files."
    )
    parser.add_argument('--val_split', type=float, default=0.2,
                   help='Fraction of "train" split to hold out for validation.'
    )
    parser.add_argument('--random_state', type=int, default=42,
                   help='Random seed for splitting.'
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
    parser.add_argument('--input_dim', type=int, default=None,
                   help='Input feature dimension (auto if unset).')
    
    parser.add_argument('--num_classes', type=int, default=None,
                   help='Number of classes.')
    
    parser.add_argument('--hidden_dims', type=str, default="128,64",
                   help='Comma-separated hidden layer sizes, e.g. "128,64".'
                   )
    parser.add_argument(
        '--lr', type=float, default=0.01, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        '--accumulation_steps', type=int, default=0,
        help="Number of steps to accumulate gradients before updating."
    )
    parser.add_argument(
        '--num_workers', type=int, default=2,
        help="Number of DataLoader worker threads."
    )
    args = parser.parse_args()

    train_list, val_list, train_label_map, val_label_map = get_file_lists(
        args.meta, sample_column='SubID', phen_column='c15x',
        val_split=args.val_split, random_state=args.random_state
    )
    train(train_list, train_label_map, val_list, val_label_map, args)


if __name__ == '__main__':
    main()
