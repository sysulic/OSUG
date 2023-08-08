import pandas as pd
import torch
import time
# import os.path as osp
from tqdm import trange
from collections import deque
from copy import deepcopy
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from ltlf2dfa.parser.ltlf import LTLfParser, LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext, LTLfOr, LTLfEventually

bop = ['&', '|', 'U']
uop = ['!', 'X', 'F', 'G']
node_op_type = {'&': 4, '|': 5, '!': 6, 'X': 7, '': 0}
binOp = [LTLfAnd, LTLfOr, LTLfUntil]
uOp = [LTLfNot, LTLfNext, LTLfEventually, LTLfAlways]
Op = [LTLfAnd, LTLfOr, LTLfUntil, LTLfNot, LTLfNext, LTLfEventually, LTLfAlways]

# node_mapper #  sub_for 0  expanded_subfor 1  atom 2  root 3  & 4  | 5  ！6  X 7 
map_com = {(0, 0, 0, 0, 0, 0, 0, 0) : 0, (0, 0, 0, 1, 1, 0, 0, 0) : 1, (0, 0, 0, 1, 0, 1, 0, 0) : 2, (0, 0, 0, 1, 0, 0, 1, 0) : 3, (0, 0, 0, 1, 0, 0, 0, 1) : 4,
         (1, 0, 0, 0, 1, 0, 0, 0) : 5, (1, 0, 0, 0, 0, 1, 0, 0) : 6, (1, 0, 0, 0, 0, 0, 1, 0) : 7, (1, 0, 0, 0, 0, 0, 0, 1) : 8, (0, 0, 1, 0, 0, 0, 0, 0) : 9,
         (0, 1, 0, 0, 1, 0, 0, 0) : 10, (0, 1, 0, 0, 0, 0, 0, 1) : 11}  # 0 Global Node
map_sim = {(0, 0, 0, 0, 0, 0, 0, 0) : 0, (0, 0, 0, 1, 1, 0, 0, 0) : 1, (0, 0, 0, 1, 0, 1, 0, 0) : 2, (0, 0, 0, 1, 0, 0, 1, 0) : 3, (0, 0, 0, 1, 0, 0, 0, 1) : 4,
         (1, 0, 0, 0, 1, 0, 0, 0) : 1, (1, 0, 0, 0, 0, 1, 0, 0) : 2, (1, 0, 0, 0, 0, 0, 1, 0) : 3, (1, 0, 0, 0, 0, 0, 0, 1) : 4, (0, 0, 1, 0, 0, 0, 0, 0) : 5,
         (0, 1, 0, 0, 1, 0, 0, 0) : 2, (0, 1, 0, 0, 0, 0, 0, 1) : 4}  # 0 Global Node

parser = LTLfParser()

class GLDataSet(InMemoryDataset):
    def __init__(self, root='data/test', name='train.json', node_map=0):
        self.root = root
        self.name = name
        if not node_map:
            self.node_map = map_com
        else:
            self.node_map = map_sim
        super().__init__(root, None, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return self.root + '/' + self.name.replace('.json', '_processed')

    @property
    def raw_file_names(self):
        return self.root + '/' + self.name

    @property
    def processed_file_names(self):
        return [self.name+'sat.pt']


    def process(self):

        print(f"Processing data from {self.root + '/' + self.name}.")
        df = pd.read_json(self.root + '/' + self.name)
        data_list = []
        for i in trange(len(df), ncols=80, desc=f'Processing'):
            data = df.loc[i]
            f_raw, y = data['inorder'], data['issat']
            if not y:
                continue
            f_inorder = f_raw
            y = 1 if y else 0    
            f_raw = parser(f_raw)

            subformulas = self.extract_subformulas(f_raw)
            expanded_subformulas = self.expand_all_subformulas(subformulas)
            x, edge_index, ver_list, u_index, atom_mask = self.ltl_to_coo(expanded_subformulas)
            y = torch.tensor(y, dtype=torch.long)
            num_node = len(ver_list)

            data_list.append(Data(x=x, edge_index=edge_index, y=y, atom_mask=atom_mask, ver_list=ver_list, num_node=num_node, u_index=u_index, f_inorder=f_inorder))
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


    def get_subformulas(self, formula):
        """
        get the op and sub-formulas in formula
        :param formula: str, LTL formula
        :return: op: LTL operator, sub1: left sub-formula sub2: right sub-formula
        """
        depth = 0
        op = ''
        is_bop = 0
        sub1 = ''
        sub2 = ''
        for i,char in enumerate(formula):
            if depth == 0 and char in uop:
                op = char
            elif depth == 0 and char in bop:
                op = char
                is_bop = 1
            elif char == '(':
                if depth == 0:
                    start = i
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0 and not is_bop:
                    sub1 = formula[start:i+1]
                if depth == 0 and is_bop:
                    sub2 = formula[start:i+1]
        return op, sub1, sub2

    def extract_subformulas(self, formula):
        """
        get and save all sub-formulas in formula
        :param formula: str, LTL formula
        :return: dict, key: no. of subformula, value: the sub-formula
        """
        subformulas = {}
        sym_sub = {}
        que = deque([(formula, "f0")])
        while que:
            f, f_id = que.pop()
            if f_id not in subformulas:
                # binary
                for i, t in enumerate(binOp): 
                    if isinstance(f,t): 
                        op = bop[i]
                        sub1 = f.formulas[0]
                        is_ocur = str(sub1)
                        is_ocur = self.rem_par(is_ocur)

                        if is_ocur not in sym_sub:
                            if isinstance(sub1, LTLfAtomic):
                                sub1 = self.rem_par(str(sub1))
                            else:  
                                que.appendleft((sub1, f'{f_id}1'))
                                sub1 = f'{f_id}1'
                        else:
                            sub1 = sym_sub[is_ocur]                     

                        if len(f.formulas) == 2:
                            sub2 = f.formulas[1]
                            is_ocur1 = is_ocur
                            is_ocur = str(sub2)
                            is_ocur = self.rem_par(is_ocur)

                            if is_ocur == is_ocur1:
                                sub2 = sub1
                            elif is_ocur not in sym_sub:
                                if isinstance(sub2, LTLfAtomic):
                                    sub2 = self.rem_par(str(sub2))
                                else:
                                    que.appendleft((sub2, f'{f_id}2'))
                                    sub2 = f'{f_id}2'
                            else: 
                                sub2 = sym_sub[is_ocur]
                            if op == 'U':
                                subformulas[f'{f_id}'] = f'({sub1}){op}({sub2})'
                                sym_sub[f'{sub1} {op} {sub2}'] = f'{f_id}'
                            else:       # invariance of  '&' '|'
                                subformulas[f'{f_id}'] = f'({sub1}){op}({sub2})'
                                sym_sub[f'{sub1} {op} {sub2}'] = f'{f_id}'
                                sym_sub[f'{sub2} {op} {sub1}'] = f'{f_id}'

                        else:
                                nf = deepcopy(f)                              
                                sub = nf.formulas

                                ids = 1         
                                sub_s = ""      
                                sym_s = ""      
                                for i, ssub in enumerate(sub):
                                    n = len(sub) - 1 
                                    if isinstance(ssub, LTLfAtomic):
                                        sub_s += f'({str(ssub)})'
                                        sym_s += f'{str(ssub)}'
                                        if i != n:
                                            sub_s += op
                                            sym_s += f' {op} '

                                    else:
                                        is_ocur = str(ssub)
                                        is_ocur = self.rem_par(is_ocur)
                                        if is_ocur not in sym_sub:
                                            que.appendleft((ssub, f'{f_id}{ids}'))
                                            sub_s += f'({f_id}{ids})'
                                            sym_s += f'{f_id}{ids}'
                                            ids += 1
                                        else:
                                            is_ocur = sym_sub[is_ocur]
                                            sub_s += f'({is_ocur})'
                                            sym_s += f'{is_ocur}'

                                        if i != n:
                                            sub_s += op
                                            sym_s += f' {op} '

                                subformulas[f'{f_id}'] = sub_s
                                sym_sub[sym_s] = f'{f_id}'
                                sym_sub[sym_s] = f'{f_id}'

                # unary
                for i, t in enumerate(uOp): 
                    if isinstance(f, t):
                        op = uop[i]
                        sub1 = f.f
                        is_ocur = str(sub1)
                        is_ocur = self.rem_par(is_ocur)

                        if is_ocur not in sym_sub:
                            if isinstance(sub1, LTLfAtomic):
                                sub1 = self.rem_par(str(sub1))
                            else:
                                que.appendleft((sub1, f'{f_id}1'))
                                sub1 = f'{f_id}1'
                        else:
                            
                            sub1 = sym_sub[is_ocur]

                        subformulas[f'{f_id}'] = f'{op}({sub1})'
                        sym_sub[f'{op}({sub1})'] = f'{f_id}'
        return subformulas

    def one_step_expansion(self, key, formula):
        """
        one step unfold the formula
        :param key: str, no. of formula
        :param formula: str, LTL formula
        :return: str, the unfolded formula
        """
        no_need = ["&", "|", "!", "X"]
        op, sub1, sub2 = self.get_subformulas(formula)
        # print(op)
        nx_ltl = ''
        if op in no_need:
            return formula, 0
        elif op == "F":
            nx_ltl = f"{sub1}|(X({key}))"
        elif op == 'U':
            nx_ltl = f"{sub2}|({sub1}&(X({key})))"
        elif op == 'G':
            nx_ltl = f"{sub1}&(X({key}))"
        elif op == '':
            return formula, 0
        else:
            raise ValueError("Invalid LTL formula")
        return nx_ltl, 1

    def expand_all_subformulas(self, subformulas):
        expanded_subformulas = {}
        for key, value in subformulas.items():
            expanded_value, is_expanded = self.one_step_expansion(key, value)
            expanded_subformulas[key] = (expanded_value, is_expanded)
        return expanded_subformulas

    def ltl_to_coo(self, formula_dic):
        """
        get the data likes coo of the LTL formula.
        :param formula_dic: key: no. of formula, value:(subformula, whether unfolded)
        """
        
        sub_dict = deepcopy(formula_dic)

        vertices = dict()   
        for key in sub_dict:
            if key == 'f0' and key not in vertices:
                vertices[key] = [3, 0]
            elif key not in vertices:
                vertices[key] = [0, 0]

        for key, value in sub_dict.items():
            op, sub1, sub2 = self.get_subformulas(value[0])
            vertices[key][1] = node_op_type[op]
            
            sub = value[0].split(op)
            if op in bop and len(sub) == 2:
                if sub1[1:-1] not in vertices:
                    vertices[sub1[1:-1]] = [value[1], 0]
                if sub2[1:-1] not in vertices:
                   vertices[sub2[1:-1]] = [value[1], 0]
            elif len(sub) > 2:
                for ssub in sub:
                    if ssub[1:-1] not in vertices:
                        vertices[ssub[1:-1]] = [0, 0]
            
            elif op in uop:
                if sub1[1:-1] not in vertices:
                    vertices[sub1[1:-1]] = [value[1], 0]
        
        tmp = deepcopy(vertices)

        for key, value in tmp.items():
            op, sub1, sub2 = self.get_subformulas(key)

            if op != '':
                vertices[key][1] = node_op_type[op]    
            if sub1 != '' and sub1[1:-1] not in vertices:
                vertices[sub1[1:-1]] = [value[0], node_op_type[self.get_subformulas(sub1[1:-1])[0]]]
            if sub2 != '' and sub2[1:-1] not in vertices:
                vertices[sub2[1:-1]] = [value[0], node_op_type[self.get_subformulas(sub2[1:-1])[0]]]

        ver_list = []
        x = []

        for key, value in vertices.items():
            ver_list.append(key)
            y = [0, 0, 0, 0, 0, 0, 0, 0]
            if key.startswith('p'):
                vertices[key] = 2
                y[2] = 1
            else:
                y[value[0]] = 1
                y[value[1]] = 1

            x.append(y)

        edge_index = [[],[]]

        for f_id, subformula in sub_dict.items():
            parent_idx = ver_list.index(f_id)
            op, sub1, sub2 = self.get_subformulas(subformula[0])

            sub = subformula[0].split(op)
            if op in bop and len(sub) >= 2:    
                for ssub in sub:
                    edge_index[0].append(parent_idx)
                    edge_index[1].append(ver_list.index(ssub[1:-1]))
                    
                    edge_index[0].append(ver_list.index(ssub[1:-1]))
                    edge_index[1].append(parent_idx)


            elif op in uop:
                edge_index[0].append(parent_idx)
                edge_index[1].append(ver_list.index(sub1[1:-1]))

                
                edge_index[0].append(ver_list.index(sub1[1:-1]))
                edge_index[1].append(parent_idx)

        
        for ver in ver_list:
            parent_idx = ver_list.index(ver)
            op, sub1, sub2 = self.get_subformulas(ver)
            if op in bop:
                edge_index[0].append(parent_idx)
                edge_index[1].append(ver_list.index(sub1[1:-1]))

                edge_index[0].append(ver_list.index(sub1[1:-1]))
                edge_index[1].append(parent_idx)


                edge_index[0].append(parent_idx)
                edge_index[1].append(ver_list.index(sub2[1:-1]))

                edge_index[0].append(ver_list.index(sub2[1:-1]))
                edge_index[1].append(parent_idx)


            if op in uop:
                edge_index[0].append(parent_idx)
                edge_index[1].append(ver_list.index(sub1[1:-1]))


                edge_index[0].append(ver_list.index(sub1[1:-1]))
                edge_index[1].append(parent_idx)

        x.append([0, 0, 0, 0, 0, 0, 0, 0])  # global node

        xx = []  
        for i in x:
            xx.append(self.node_map[tuple(i)])

        ver_list.append('U')       
        u_index = len(ver_list) - 1
        for i in range(len(ver_list)-1):
            edge_index[0].append(i)
            edge_index[1].append(u_index)

            edge_index[0].append(u_index)
            edge_index[1].append(i)

        x = torch.tensor(x, dtype=torch.float32)
        atom_mask = x[:, 2] 
        x = torch.tensor(xx, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return x, edge_index, ver_list, u_index, atom_mask
    
    def rem_par(self, formula):
        if formula.startswith("(") and formula.endswith(")"):
            counter = 0
            for i in range(1, len(formula)-1):
                if formula[i] == "(":
                    counter += 1
                elif formula[i] == ")":
                    counter -= 1
                if counter < 0:
                    break
            if counter == 0:
                formula = formula[1:-1]
                formula = self.rem_par(formula)
        return formula
    
    def download(self):
        pass
 

if __name__ == '__main__':

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    start = time.time()
    dataset = GLDataSet(root='data/LTLSATUNSAT-{and-or-not-F-G-X-until}-100-random/[100-200)/', name='debug.json')

    end_time_GLDataSet = time.time()
    time_GLDataSet = end_time_GLDataSet - start
    loader = DataLoader(dataset, batch_size=4,shuffle=False)
    end = time.time()
    time_DataLoader = end - end_time_GLDataSet
    elapsed = end - start
    for data in loader:
        print(data)
        print(data.x)
        print(data.edge_index)
        print(data.y)

    print(f'number of example: {len(dataset)}.')
    print(f'cost time of GLDataSet: {time_GLDataSet}.')
    print(f'cost time of DataLoader: {time_DataLoader}')
    print(f'cost time of total: {elapsed}')
