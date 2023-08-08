import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model_sc import GNN
from data_sc import GLDataSet
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

def seed_everywhere(seed):
    torch.manual_seed(seed)     # cpu
    torch.cuda.manual_seed(seed)    # gpu
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def plotCurve(valLosses, model_dir):
    plt.figure()
    plt.xlabel('Training step')
    plt.ylabel('Validation Loss')
    plt.title("Learning Curve")
    plt.grid()
    plt.plot(range(1, len(valLosses) + 1), valLosses, 'o-', color="r")
    plt.savefig(join(model_dir, 'train_curve.jpg'))
    # plt.show()

def main(args):

    seed_everywhere(seed=args.seed)
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_dir_name = f"{time_str}_eds_{args.eds}_lr_{args.lr}_hc_{args.hc}_bs_{args.bs}_nl_{args.nl}_ear_{args.ear}_seed_{args.seed}"
    model_dir = join('model', model_dir_name)
    
    os.makedirs(model_dir)
    print(f"Save model at {model_dir}.")
    log_dir = join('log', model_dir_name)
    os.makedirs(log_dir)
    summaryWriter = SummaryWriter(log_dir)

    device = torch.device(args.dv if torch.cuda.is_available() else "cpu")

    train_dataset = GLDataSet(root=args.dt, name=args.trd, node_map=args.nm)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    dev_dataset = GLDataSet(root=args.dt, name=args.vd, node_map=args.nm)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bs, shuffle=True)

    model = GNN(
                in_channels= args.ic,
                hidden_channels=args.hc,
                num_layers=args.nl,
                out_channels=args.oc,
                embedding_size=args.eds).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss() 

    best_val_loss = float('inf')
    best_model = None
    total_step = 0
    val_loss = []
    for epoch in range(args.e):
        model.train()

        total_loss = 0

        for data in tqdm(train_loader, ncols=100, desc=f'Epoch {epoch + 1}/{args.e}'):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)

            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_step += 1

            summaryWriter.add_scalar("training step loss", loss.cpu().item(), total_step)
    
            total_loss += loss.item()

        av_loss = total_loss / len(train_loader)

        model.eval()

        total_loss = 0

        total_correct = 0
        for data in tqdm(dev_loader, ncols=100, desc=f'Epoch {epoch + 1}/{args.e}'):
            data = data.to(device)
            with torch.no_grad():
                out = model(data)
            
            loss = criterion(out, data.y)

            total_loss += loss.item()

            pred = out.argmax(dim=1)
            
            total_correct += (pred == data.y).sum().item()
            
        dev_loss, dev_acc = total_loss / len(dev_loader), total_correct / len(dev_dataset)

        summaryWriter.add_scalar("training loss", av_loss, total_step)
        summaryWriter.add_scalar("validation loss", dev_loss, total_step)
        summaryWriter.add_scalar("validation acc", dev_acc, total_step)
        val_loss.append(dev_loss)
        
        if dev_loss < best_val_loss:
            best_model = model
            best_val_loss = dev_loss
            bestPath = join(model_dir, 'step{%d}-lr{%.4f}-early{%d}-loss{%.2f}-acc{%.2f}.pth' % (total_step, args.lr, args.ear, dev_loss, dev_acc))
            torch.save(best_model.state_dict(), bestPath)
            print(f"Best model save at {bestPath}.")
            epsilon = 0
        else:
            epsilon += 1
            if epsilon >= args.ear:
                break
        print(f"Epoch: {epoch+1}/{args.e}, Train Loss: {av_loss:.8f}, Dev Loss: {dev_loss:.8f}, Dev Acc: {dev_acc:.8f}")
        if epsilon >= args.ear:
            print(f"Done due to early stopping.")
            break
    
    plotCurve(val_loss, model_dir)

        

if __name__ == '__main__':
    parser = ArgumentParser(description='Train LTL embedding')

    parser.add_argument('--e', type=int, default=256, help="epochs") 
    parser.add_argument('--nl', type=int, default=10, help="number of layers") 
    parser.add_argument('--eds', type=int, default=256, help="var embedding size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--bs', type=int, default=512, help="batch size")
    parser.add_argument('--ear', type=int, default=200, help="early stop after ear epochs")
    parser.add_argument('--wd', type=float, default=0, help="weight decay rate")
    parser.add_argument('--dv', type=str, default='cuda:0', help="device")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--nm', type=int, default=0, help="node map class")

    parser.add_argument('--ic', type=int, default=12, help="node features")
    parser.add_argument('--hc', type=int, default=512, help="hidden dimension")
    parser.add_argument('--oc', type=int, default=2, help="number of class")

    parser.add_argument('--dt', type=str, default='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/', help="data dir")
    parser.add_argument('--trd', type=str, default='train.json', help="training dataset")
    parser.add_argument('--vd', type=str, default='dev.json', help="validation dataset")

    args = parser.parse_args()
    main(args)
