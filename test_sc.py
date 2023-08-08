import torch
import sys
import os
import time
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from os.path import join
from tqdm import tqdm
from model_sc import GNN
from data_sc import GLDataSet

sys.setrecursionlimit(10**5)

def main(args):
    
    total_time = 0

    device = torch.device(args.dv if torch.cuda.is_available() else "cpu")

    test_dataset = GLDataSet(root=args.dt, name=args.td, node_map=args.nm)

    best_model_path = join(args.trp, args.sbm)

    print(f"Testing GNN on {args.dt}_{args.td}")
    model = GNN(
                in_channels= args.ic,
                hidden_channels=args.hc,
                num_layers=args.nl,
                out_channels=args.oc,
                embedding_size=args.eds).to(device)

    model.load_state_dict(torch.load(best_model_path))

    TP, FP, TN, FN = 1e-6, 1e-6, 1e-6, 1e-6
    res = []

    model.eval()

    for data in tqdm(test_dataset, ncols=100, desc='Testing'):
        data = data.to(device)
        expect = data.y
        local_start = time.time()
        with torch.no_grad():
            out = model(data)
            local_end = time.time()
        pred = out.argmax(dim=1)
        if expect and expect == pred:
            TP += 1
        elif expect and expect != pred:
            FN += 1
        elif not expect and expect == pred:
            TN += 1
        else:
            FP += 1

        res.append((expect, pred, local_end - local_start))
        total_time += local_end - local_start

    Acc, Pre, Rec, F1 = (TP + TN) / (TP + TN + FP + FN), \
        TP / (TP + FP), \
        TP / (TP + FN), \
        (2 * TP) / (2 * TP + FP + FN)

    print('Average test time: %.4f' % (total_time / (TP + TN + FP + FN)))
    print('Total test time: %.4f' % (total_time))
    print('Total   : (TP, TN, FP, FN) = (%d, %d, %d, %d)' % (TP, TN, FP, FN))
    print('Total   : (Acc, P, R, F1) = (%.4f, %.4f, %.4f, %.4f)' % (Acc, Pre, Rec, F1))

    res_dir = join(args.trp, args.dt.split('/')[-1], args.td.split('.json')[0])
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    with open(join(res_dir, 'result.txt'), "w") as f:
        print('Average test time: %.4f' % (total_time / (TP + TN + FP + FN)), file=f)
        print('Total test time: %.4f' % (total_time), file=f)
        print('Total   : (TP, TN, FP, FN) = (%d, %d, %d, %d)' % (TP, TN, FP, FN), file=f)
        print('Total   : (Acc, P, R, F1) = (%.4f, %.4f, %.4f, %.4f)' % (Acc, Pre, Rec, F1), file=f)
        print(Acc, Pre, Rec, F1, total_time, sep='\t', file=f)

        for l in tqdm(res):
            # print(l[0], "sat" if l[1] else "unsat", l[2], sep='\t', file=f)
            print("sat" if l[0] else "unsat", f'predict: {"sat" if l[1] else "unsat"}', f'time: {l[2]}', sep='\t', file=f)
    


if __name__ == '__main__':
    parser = ArgumentParser(description='Test GLTL')
    parser.add_argument('--nl', type=int, default=10, help="number of layers") 
    parser.add_argument('--eds', type=int, default=256, help="var embedding size")
    parser.add_argument('--bs', type=int, default=256, help="batch size")
    parser.add_argument('--dv', type=str, default='cuda:0', help="device")
    parser.add_argument('--nm', type=int, default=0, help="node map class")

    parser.add_argument('--ic', type=int, default=12, help="node features")
    parser.add_argument('--hc', type=int, default=512, help="hidden dimension")
    parser.add_argument('--oc', type=int, default=2, help="number of class")

    parser.add_argument('--dt', type=str, required=True, help="data dir")
    parser.add_argument('--sbm', type=str, required=True, help="saved best model")
    parser.add_argument('--trp', type=str, required=True, help="test record path")

    args = parser.parse_args()
    main(args)
