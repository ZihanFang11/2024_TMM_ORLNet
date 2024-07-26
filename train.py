
import torch.nn as nn
import warnings
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from util.loadMatData import load_data_semi,features_to_Lap,features_T_to_Lap
from ORLNet_model import CombineNet
from sklearn import metrics


def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    MAF1 = metrics.f1_score(labels_true, labels_pred, average='macro')
    MIF1 = metrics.f1_score(labels_true, labels_pred, average='micro')
    return ACC, P, R, MAF1, MIF1
def train(args, device, feature_list,  lap_Z, lap_G, labels):

    print("*" * 40 +  "gamma:{}, gamma2:{}, block:{}, epoch:{}, thre1:{}, thre2:{},lr:{}, lambda:{},alpha:{}\n".format(
                        args.gamma,  args.gamma2, args.block, args.epoch,
                        args.thre1, args.thre2,
                        args.lr, args.lamb, args.alpha))
    ACC_list = []
    f1_macro_list = []
    f1_micro_list = []
    P_list=[]
    R_list=[]
    time_list = []
    for kk in range(args.resp):
        model = CombineNet(n, n_feats, n_view, n_classes, args.block, args.thre1, args.thre2, args.alpha, args.lamb, device,
                           args.fusion_type).to(device)

        criterion=nn.CrossEntropyLoss()
        criterion2=nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ACC = 0
        F1_macro = 0
        F1_micro = 0

        loss_list = []
        acc_list = []
        f1_macro_list = []
        f1_micro_list = []

        begin_time = time.time()
        with tqdm(total=args.epoch, desc="Training") as pbar:
            for epoch in range(args.epoch):
                model.train()
                output,Zv_List = model(feature_list, lap_Z, lap_G)

                loss_sup = criterion2(output[idx_train], labels[idx_train])
                loss_sup2=0
                for i in range(n_view):
                    loss_sup2+=criterion(Zv_List[i][idx_train], labels[idx_train])
                loss = args.gamma * loss_sup + args.gamma2 * loss_sup2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    output,Zv_List= model(feature_list, lap_Z, lap_G)
                    pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                    ACC, P, R, F1_macro, F1_micro = get_evaluation_results(labels.cpu().detach().numpy()[idx_test],
                                                                     pred_labels[idx_test])

                    print({'Loss': '{:.6f}'.format(loss.item()),
                           'ACC': '{:.2f}'.format(ACC * 100)}),
                    loss_list.append(loss.item())
                    f1_macro_list.append(F1_macro * 100)
                    f1_micro_list.append(F1_micro * 100)

                    acc_list.append(ACC*100)
                    pbar.update(1)


        print("------------------------")
        print("ACC:   {:.2f}".format(ACC * 100))
        print("F1_macro:   {:.2f}".format(F1_macro * 100))
        print("F1_micro:   {:.2f}".format(F1_micro * 100))
        print("------------------------")
        cost_time = time.time() - begin_time
        ACC_list.append(ACC)
        f1_macro_list.append(F1_macro)
        f1_micro_list.append(F1_micro)
        P_list.append(P)
        R_list.append(R)
        time_list.append(cost_time)


    acc = str(round(np.mean(ACC_list) * 100, 2)) + " (" + str(round(np.std(ACC_list) * 100, 1)) + ")"
    P = str(round(np.mean(P_list) * 100, 2)) + " (" + str(round(np.std(P_list) * 100, 1)) + ")"
    R = str(round(np.mean(R_list) * 100, 2)) + " (" + str(round(np.std(R_list) * 100, 1)) + ")"

    f1_macro = str(round(np.mean(f1_macro_list) * 100, 2)) + " (" + str(
        round(np.std(f1_macro_list) * 100, 2)) + ")"
    f1_micro = str(round(np.mean(f1_micro_list) * 100, 2)) + " (" + str(
        round(np.std(f1_micro_list) * 100, 2)) + ")"
    Runtime_mean_std = str(round(np.mean(time_list), 2)) + " (" + str(
        round(np.std(time_list), 2)) + ")"
    if args.save_total_results:
        with open(args.save_file, "a") as f:
            f.write(
                "fusion:{}, gamma:{},gamma2:{} block:{}, epoch:{}, thre1:{}, thre2:{},lr:{}, lambda:{},alpha:{}, ratio:{}\n".format(
                    args.fusion_type,args.gamma,args.gamma2 ,args.block, args.epoch,
                    args.thre1, args.thre2,
                    args.lr, args.lamb, args.alpha, args.ratio))
            f.write("{}:{}\n".format(data, dict(
                zip(['acc', 'F1_macro', 'F1_micro','precision', 'recall', 'time', ], [acc, f1_macro, f1_micro, P ,R,  Runtime_mean_std]))))


import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="")

    parser.add_argument("--save_results", action='store_true', default=False, help="Save experimental result.")
    parser.add_argument("--resp", type=int, default=1)
    parser.add_argument("--save_total_results", action='store_true', default=True, help="Save experimental result.")
    parser.add_argument("--dropout", type=float, default=0., help="dropout")

    parser.add_argument("--nhid", type=int, default=32, help="Number of hidden dimensions")
    parser.add_argument("--path", type=str, default="./data/", help="Path of datasets.")
    parser.add_argument("--save_file", type=str, default="result.txt")

    parser.add_argument("--use_seed", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=20, help="Random seed, default is 42.")
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument("--cuda_devices", default="0", type=str, required=False)
    parser.add_argument("--ratio", type=float, default=0.1, help="Number of labeled samples per classes")
    parser.add_argument("--weight_decay", type=float, default=0.15, help="Weight decay")

    parser.add_argument("--fusion_type", type=str, default="weight")
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--gamma2', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--block', type=int, default=3, help='block')
    parser.add_argument('--thre1', type=float, default=0.1)
    parser.add_argument('--thre2', type=float, default=0.1)
    parser.add_argument('--lamb', type=float, default=1, help='lambda')
    parser.add_argument('--alpha', type=float, default=1, help='alpha')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parameter_parser()
    seed = args.seed
    path = args.path
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    args.device = device

    # tab_printer(args)

    dataset_dict = {1: 'MIRFlickr', 2: 'HW', 3: 'NUSWide20k', 4: 'NUSWIDE',  5: 'WebKB_texas', 6: 'Youtube'}

    block_data = { 'HW':3,  'Hdigit':2, 'NUSWIDE':3, 'WebKB_texas':3, 'Youtube':3 , 'MIRFlickr':3,'NUSWide20k':5}


    select_dataset = [1,2,3,4,5,6]

    for data_id in select_dataset:

        data = dataset_dict[data_id]
        args.block = block_data[data]
        if data=='NUSWIDE' or data=='WebKB_texas':
            args.gamma=10

        if data == 'Youtube':
            args.alpha=10
        print("========================",data)

        features, labels, idx_train, idx_test = load_data_semi(args, data)

        print(len(idx_train), len(idx_test))
        n_view = len(features)
        n_feats = [x.shape[1] for x in features]
        n = features[0].shape[0]
        n_classes = len(np.unique(labels))
        labels = labels.to(device)

        print(data,n,n_view,n_feats)

        feature_list=[]
        for i in range(n_view):
            feature_list.append(torch.from_numpy(features[i]/1.0).float().to(device))




        lap_Z = features_to_Lap(data,features, int(n / n_classes))
        for i in range(n_view):
            lap_Z[i] =torch.from_numpy(lap_Z[i]).float().to(device)

        lap_G = features_T_to_Lap(data, [_.T for _ in features], 0.5)
        for i in range(n_view):
            lap_G[i] = torch.from_numpy(lap_G[i]).float().to(device)


        if args.fix_seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)

        train(args, device, feature_list, lap_Z, lap_G, labels)



