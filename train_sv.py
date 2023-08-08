import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model_sv import GNN
from data_sv import GLDataSet
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from copy import deepcopy
from partial_logic_checker import check
from torch.distributions.categorical import Categorical
import json

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

class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer=[0]*capacity
        self.buffer_count=0
        self.capacity=capacity
        self.batch_size=batch_size

    def __len__(self):
        return self.buffer_count

    def push(self, *transition):
        self.buffer[self.buffer_count%self.capacity]=transition
        self.buffer_count+=1

    def sample(self):
        return [self.buffer[x] for x in np.random.choice(self.capacity, self.batch_size, replace=False)]

def model_check(trace, loop_start, ltl: str, verlist, cut_time):
    vocab=[f'p{i}' for i in range(100)]
    trace_list=[]
    for t in range(cut_time):
        sub_list=[]
        for i, p in enumerate(verlist):
            if p in vocab and trace[i][t]:
                sub_list.append(p)
        trace_list.append(sub_list)
    trace=(trace_list, loop_start)
    return check(ltl,trace,vocab)

def make_action(model, x, edge_index, trace, batch, u_index, inloop, predict_time, atom_mask, device, epsilon=None, mode='test'):
    x=x.to(device)
    edge_index=edge_index.to(device)
    trace=trace.to(device)
    batch=batch.to(device)
    u_index=u_index.to(device)
    inloop=inloop.to(device)
    predict_time=torch.tensor([predict_time], dtype=torch.long, device=(device))
    atom_mask=atom_mask.to(device)

    with torch.no_grad():
        predict_trace, predict_inloop=model(x, edge_index, trace, batch, u_index, inloop, predict_time, atom_mask)
        predict_trace=predict_trace.detach()
        predict_inloop=predict_inloop.detach()

    if mode=='test' or np.random.random()>epsilon:
        return predict_trace.argmax(dim=-1).cpu(), predict_inloop.argmax(dim=-1).cpu()
    else:
        predict_trace*=0
        predict_inloop*=0
        return Categorical(probs=predict_trace.softmax(dim=-1)).sample().cpu(), Categorical(probs=predict_inloop.softmax(dim=-1)).sample().cpu()

def learn(eval_model, target_model, replay_buffer, device, optimizer):
    sample=replay_buffer.sample()
    x, edge_index, trace, u_index, inloop, predict_time, atom_mask, reward, predict_trace, predict_inloop, done=zip(*sample)
    batch_size=len(sample)
    node_size=[0]*batch_size
    for i in range(1, batch_size):
        node_size[i]=node_size[i-1]+atom_mask[i-1].shape[0]
    x=torch.cat(x, dim=0).to(device)
    edge_index=torch.cat([edge+node_size[i] for i, edge in enumerate(edge_index)], dim=1).to(device)
    trace=torch.cat(trace, dim=0).to(device)
    batch=torch.tensor([i for i in range(batch_size) for j in range(atom_mask[i].shape[0])], dtype=torch.long).to(device)
    inloop=torch.cat(inloop, dim=0).to(device)
    u_index=torch.cat([index+node_size[i] for i, index in enumerate(u_index)], dim=0).to(device)
    atom_mask=torch.cat(atom_mask, dim=0).to(device)
    predict_time=torch.tensor(predict_time).to(device)
    predict_trace=torch.cat(predict_trace, dim=0).to(device)
    predict_inloop=torch.cat(predict_inloop, dim=0).to(device)
    reward=torch.tensor(reward, dtype=torch.float).to(device)
    done=torch.tensor(done, dtype=torch.float).to(device)
    
    next_trace=trace.detach()
    next_trace[torch.arange(batch.shape[0], dtype=torch.long, device=device), predict_time[batch]]=predict_trace
    next_inloop=inloop.detach()
    next_inloop[torch.arange(batch.shape[0], dtype=torch.long, device=device), predict_time[batch]]=predict_inloop[batch]

    eval_model.train()
    trace_score, inloop_score=eval_model(x, edge_index, trace, batch, u_index, inloop, predict_time, atom_mask)
    with torch.no_grad():
        target_trace_score, target_inloop_score=target_model(x, edge_index, next_trace, batch, u_index, next_inloop, (predict_time+1).clamp(min=0, max=4), atom_mask)
        target_trace_score=target_trace_score.detach()
        target_inloop_score=target_inloop_score.detach()
    
    sum_matrix=torch.zeros((batch_size, batch.shape[0]), dtype=torch.float, device=device)
    sum_matrix[batch, torch.arange(batch.shape[0], dtype=torch.long, device=device)]=1
    
    q_eval=sum_matrix.matmul(trace_score.gather(-1, predict_trace.unsqueeze(dim=-1)).squeeze(dim=-1)*atom_mask)\
            +inloop_score.gather(-1, predict_inloop.unsqueeze(dim=-1)).squeeze(dim=-1)
    q_target=reward+(1-done)*(sum_matrix.matmul(target_trace_score.max(dim=-1)[0]*atom_mask)+target_inloop_score.max(dim=-1)[0])
    loss=F.mse_loss(q_eval, q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    eval_model.eval()

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
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    dev_dataset = GLDataSet(root=args.dt, name=args.vd, node_map=args.nm)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=True)

    eval_model = GNN(
                in_channels= args.ic,
                hidden_channels=args.hc,
                num_layers=args.nl,
                out_channels=args.oc,
                embedding_size=args.eds,
                k=args.mtl).to(device)
    target_model=GNN(
                in_channels= args.ic,
                hidden_channels=args.hc,
                num_layers=args.nl,
                out_channels=args.oc,
                embedding_size=args.eds,
                k=args.mtl).to(device)

    optimizer = torch.optim.Adam(eval_model.parameters(), lr=args.lr, weight_decay=args.wd)
    replay_buffer=ReplayBuffer(args.capacity, args.bs)

    best_model = None
    total_step = 0
    best_total_reward=-1
    total_reward=0
    epsilon=args.epsilon_start
    eval_model.eval()
    target_model.eval()
    learn_step=0
    for epoch in range(args.e):
        for data in tqdm(train_loader, ncols=100, desc=f'Epoch {epoch + 1}/{args.e}'):

            x, edge_index, u_index, in_order, atom_mask, batch, ver_list = data.x, data.edge_index, data.u_index.unsqueeze(dim=-1), data.f_inorder[0], data.atom_mask, data.batch, data.ver_list[0]
            trace=torch.zeros((x.shape[0], args.mtl), dtype=torch.long)
            inloop=torch.zeros((x.shape[0], args.mtl), dtype=torch.long)
            eposide_reward=0
            for t in range(args.mtl):
                predict_trace, predict_inloop=make_action(eval_model, x, edge_index, trace, batch, u_index, inloop, t, atom_mask, device, epsilon=epsilon, mode='train')
                
                next_trace=trace.detach()
                next_trace[:, t]=predict_trace
                next_inloop=inloop.detach()
                next_inloop[:, t]=predict_inloop[batch]

                flag=model_check(next_trace, -1, in_order, ver_list, t+1)
                reward=int(flag)
                reward+=(0 if t==0 else (next_inloop[0, t-1]<=next_inloop[0, t]))
                if t==args.mtl-1:
                    loop_start=next_inloop[0].argmax() if next_inloop[0].max()==1 else args.mtl-1
                    reward+=(int(next_inloop[0, t]==1)+int(model_check(next_trace, loop_start, in_order, ver_list, t+1)))*10

                eposide_reward+=reward
                replay_buffer.push(x, edge_index, trace, u_index, inloop, t, atom_mask, reward, predict_trace, predict_inloop, (not flag) or (t==args.mtl-1))
                trace=next_trace
                inloop=next_inloop

                if len(replay_buffer)>=args.capacity:
                    if (learn_step+1)%args.replace_target_freq==0:
                        target_model.load_state_dict(eval_model.state_dict())
                    
                    if epsilon>args.epsilon_end:
                        epsilon*=args.epsilon_decay

                    learn(eval_model, target_model, replay_buffer, device, optimizer)

                if not flag:
                    break
            total_reward+=eposide_reward

            total_step += 1

            summaryWriter.add_scalar("training step reward", eposide_reward, total_step)

            if total_step % 100==0:
                print(f'100 reward {total_reward}')
                if total_reward > best_total_reward:
                    best_model = eval_model
                    best_total_reward = total_reward
                    bestPath = join(model_dir, 'step{%d}-lr{%.4f}-early{%d}-eposide_reward{%.2f}.pth' % (total_step, args.lr, args.ear, best_total_reward))
                    torch.save(best_model.state_dict(), bestPath)
                    print(f"Best model save at {bestPath}.")
                total_reward=0
            

def test(args):
    seed_everywhere(seed=args.seed)
    
    model_dir_name = args.model_file
    model_dir = join('model', model_dir_name)

    device = torch.device(args.dv if torch.cuda.is_available() else "cpu")

    test_dataset = GLDataSet(root=args.dt, name=args.td, nnfeatures=args.ic, max_trace_len=args.mtl)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = GNN(
                in_channels= args.ic,
                hidden_channels=args.hc,
                num_layers=args.nl,
                out_channels=args.oc,
                embedding_size=args.eds,
                k=args.mtl)
    model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    model.to(device)

    vocab=[f'p{i}' for i in range(100)]
    acc=0
    pred=[]
    for data in tqdm(test_loader, ncols=100):
        model.eval()
        x, edge_index, u_index, in_order, atom_mask, batch, ver_list = data.x, data.edge_index, data.u_index.unsqueeze(dim=-1), data.f_inorder[0], data.atom_mask, data.batch, data.ver_list[0]
        trace=torch.zeros((x.shape[0], args.mtl), dtype=torch.long)
        inloop=torch.zeros((x.shape[0], args.mtl), dtype=torch.long)

        for t in range(args.mtl):
            predict_trace, predict_inloop=make_action(model, x, edge_index, trace, batch, u_index, inloop, t, atom_mask, device, mode='test')
            
            next_trace=trace.detach()
            next_trace[:, t]=predict_trace
            next_inloop=inloop.detach()
            next_inloop[:, t]=predict_inloop[batch]

            flag=model_check(next_trace, -1, in_order, ver_list, t+1)

            trace=next_trace
            inloop=next_inloop
            if not flag:
                break
        
        loop_start=inloop[0].argmax().item() if inloop[0].max()==1 else args.mtl-1
        acc+=int(model_check(trace, loop_start, in_order, ver_list, args.mtl))

        trace_list=[]
        for t in range(trace.shape[1]):
            sub_list=[]
            for i, p in enumerate(ver_list):
                if p in vocab and trace[i][t]:
                    sub_list.append(p)
            trace_list.append(sub_list)
        pred.append({'predict_trace': trace_list, 'loop_start': loop_start})

        if len(pred)%100==0:
            print('acc: ', acc/len(pred))
    
    with open(join('result', model_dir_name.split('/')[-1][:-3]+'json'), 'w') as f:
        json.dump(pred, fp=f, indent=4)
    
    print('acc: ', acc/len(test_dataset))
        

if __name__ == '__main__':
    parser = ArgumentParser(description='Train LTL embedding')

    parser.add_argument('--e', type=int, default=256, help="epochs") 
    parser.add_argument('--nl', type=int, default=3, help="number of layers") 
    parser.add_argument('--eds', type=int, default=256, help="var embedding size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--bs', type=int, default=128, help="batch size")
    parser.add_argument('--ear', type=int, default=200, help="early stop after ear epochs")
    parser.add_argument('--mtl', type=int, default=5, help="max trace length")
    parser.add_argument('--wd', type=float, default=0, help="weight decay rate")
    parser.add_argument('--dv', type=str, default='cuda:0', help="device")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--capacity', type=int, default=1000)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=float, default=0.999)
    parser.add_argument('--is_train', type=int, required=True)
    parser.add_argument('--model_file', type=str, default='2023_05_24_21_53_18_eds_256_lr_0.001_hc_512_bs_128_nl_3_ear_200_seed_1234/step{4600}-lr{0.0010}-early{200}-eposide_reward{2327.00}.pth')
    parser.add_argument('--replace_target_freq', type=int, default=100)
    parser.add_argument('--nm', type=int, default=0, help="node map class")

    parser.add_argument('--ic', type=int, default=12, help="node features")
    parser.add_argument('--hc', type=int, default=512, help="hidden dimension")
    parser.add_argument('--oc', type=int, default=2, help="number of class")

    parser.add_argument('--dt', type=str, default='../nx_gltl/data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/', help="data dir")
    parser.add_argument('--trd', type=str, default='train.json', help="training dataset")
    parser.add_argument('--vd', type=str, default='dev.json', help="validation dataset")
    parser.add_argument('--td', type=str, default='test.json', help='test dataset')

    args = parser.parse_args()
    if args.is_train:
        main(args)
    else:
        test(args)
