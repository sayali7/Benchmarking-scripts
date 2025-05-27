import os
import sys
import argparse

from utils_dist import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
#import torch.multiprocessing as mp
#from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def train(train_list, train_label_map, args):

    print ("Created processes...")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(rank, world_size)
    print (f"Process rank: [{rank}]")
    
    dist.barrier()
    
    sharded_train_file_list = get_sharded_file_list(train_list, rank, world_size)
    print(f"Rank {rank} got {len(sharded_train_file_list)} files") 

    dataset = EmbDataset(sharded_train_file_list, args.data_path, train_label_map, file_format="parquet")
    print(f"Rank {rank} dataset length: {len(dataset)}") 

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    epochs = args.epochs
    accumulation_steps = args.accumulation_steps
    lr = args.lr
    sample_data, _, _ = dataset[0]
    input_dim = sample_data.shape[0]
    num_classes = len(set(train_label_map.values()))

    model = MLP(input_dim, num_classes).to(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss(reduction='none').to(rank)
    optimizer = optim.Adam(ddp_model.parameters(), lr=lr)
    
    print ("Training started...")
    # Training loop.
    try:
        for epoch in range(epochs):
            sampler.set_epoch(epoch)  # Ensure shuffling is different each epoch.
            ddp_model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            for i,batch in enumerate(tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)):
                batch_x, batch_y, batch_weights = batch

                batch_x = batch_x.to(rank)
                batch_y = batch_y.to(rank)
                batch_weights = batch_weights.to(rank)
                
                outputs = ddp_model(batch_x)
                loss = criterion(outputs, batch_y)
                # print (batch_weights)
                weighted_loss = ((loss * batch_weights).mean())/accumulation_steps
                #print (weighted_loss)
                if not torch.isfinite(weighted_loss):
                    print(f"[Rank {rank}] Non-finite loss encountered at epoch {epoch}, batch {i}")
                    dist.barrier()
                    cleanup()
                    sys.exit(1)

                try:
                    weighted_loss.backward()
                except Exception as e:
                    print(f"[Rank {rank}] Exception in backward: {e}")
                    dist.barrier()
                    cleanup()
                    sys.exit(1)


                torch.cuda.synchronize(device=rank)
                # if accumulation_steps>0 and (i + 1) % accumulation_steps == 0:
                #     optimizer.step()     
                #     optimizer.zero_grad()
                # else:
                optimizer.step()     
                torch.cuda.synchronize(device=rank)
                optimizer.zero_grad()
                
                running_loss += weighted_loss.item()

            # if (i + 1) % accumulation_steps != 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            
            # Only rank 0 prints the average loss.
            if rank == 0:
                print (running_loss)
                avg_loss = running_loss / len(dataloader)
                print(f"[{rank}] Epoch [{epoch+1}/{epochs}], Loss: {avg_loss}")
            if rank == 1:
                print (running_loss)
                avg_loss = running_loss / len(dataloader)
                print(f"[{rank}] Epoch [{epoch+1}/{epochs}], Loss: {avg_loss}")
            dist.barrier()

    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        cleanup()
        sys.exit(1)
    finally:  
        # if dist.is_initialized():
        #     dist.barrier()
        #dist.barrier()
        cleanup()



def main():
    parser = argparse.ArgumentParser(description="Load training data from a list of file names.")
    parser.add_argument(
        '--train_test_meta', type=str,
        default="/media/sayalialatkar/T9/Sayali/FoundationModels/scBrainLLM/phenotype_classification_files/c15x_split_seed42.pkl", 
        help="Path to the file that contains the list of training/test file names."
    )
    parser.add_argument(
        '--data_path', type=str, default=".",
        help="Path to data"
    )
    parser.add_argument(
        '--sample_column', type=str, default="SubID",
        help="Column name for samples in train_test_meta files"
    )
    parser.add_argument(
        '--phenotype_column', type=str, default="c15x",
        help="Phenotype column name for in train_test_meta files"
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help="Training epochs"
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help="Batch-size"
    )
    parser.add_argument(
        '--input_dim', type=int, default=32,
        help="Input dimension of data"
    )
    parser.add_argument(
        '--num_classes', type=int, default=2,
        help="Number of classes"
    )
    parser.add_argument(
        '--lr', type=float, default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        '--accumulation_steps', type=int, default=10,
        help="Gradient accumulatoin steps"
    )
    
    args = parser.parse_args()
    
    
    train_list, test_list, train_label_map, test_label_map = get_file_lists(args.train_test_meta, args.sample_column, args.phenotype_column)
    train(train_list,train_label_map, args)
    print ("Training complete...")

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     train()
