
from copy import deepcopy
import sys
from ltlf2dfa.parser.ltlf import LTLfParser, LTLfAnd, LTLfUntil, LTLfNot, LTLfAlways, LTLfAtomic, LTLfNext, LTLfOr, LTLfEventually, LTLfImplies, LTLfRelease

proof=''
proofcnt=0
printproof=False
len_cache={}
switch_cnt = [0] * 100

def formula_len(f,cache):
    if cache.get(f,-1)!=-1:
        return cache[f]
    if isinstance(f,str):
        cache[f] = 1
        return 1
    else:
        cnt=0
        for i in f:
            cnt+=formula_len(i,cache)
        cache[f]=cnt
        return cnt

def get_switch_cnt():
    global switch_cnt
    return switch_cnt

def checkLTL(f, f_wait, t, trace, loop_start, vocab, c={}, v=False, formula_start=0, expect_val=1, proof_dic={}):
    """ Checks satisfaction of a LTL formula on an execution trace

        NOTES:
        * This works by using the semantics of LTL and forward progression through recursion
        * Note that this does NOT require using any off-the-shelf planner

        ARGUMENTS:
            f       - an LTL formula (must be in TREE format using nested tuples
                      if you are using LTL dict, then use ltl['str_tree'])
            t       - time stamp where formula f is evaluated
            trace   - execution trace (a dict containing:
                        trace['name']:    trace name (have to be unique if calling from a set of traces)
                        trace['trace']:   execution trace (in propositions format)
                        trace['plan']:    plan that generated the trace (unneeded)
            vocab   - vocabulary of propositions
            c       - cache for checking LTL on subtrees
            v       - verbosity

        OUTPUT:
            satisfaction  - true/false indicating ltl satisfaction on the given trace
    """
    global len_cache
    global switch_cnt

    if t==len(trace['trace']):
        if loop_start==-1:
            return (expect_val,(t,(formula_start,formula_start+formula_len(f,len_cache)-1),expect_val))
        else:
            t=loop_start

    proof_node=(t,(formula_start,formula_start+formula_len(f,len_cache)-1),expect_val)
    sub_node_list=[]
    sub_node_mode='or'

    if v:
        print('\nCurrent t = ' + str(t))
        print('Current f =', f)

    # Check if first operator is a proposition
    if type(f) is str and f in vocab:
        # proofcnt-=1
        switch_cnt[0]+=1
        return (f in trace['trace'][t], proof_node)
    if type(f) is str and f=='0':
        switch_cnt[0] += 1
        # proofcnt-=1
        return (False, proof_node)
    if type(f) is str and f=='1':
        switch_cnt[0] += 1
        # proofcnt-=1
        return (True, proof_node)

    # Check if sub-tree info is available in the cache
    key = (f, t, trace['name'])
    if c is not None:
        if key in c:
            if v: print('Found subtree history')
            # proofcnt-=1
            return (c[key],proof_node)
    if key in f_wait:
        f_wait[key]+=1
    else:
        f_wait[key]=1

    # Check for standard logic operators
    f0case=0
    if f[0] in ['not', '!']:
        value,sub_node = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v, formula_start+1, 1-expect_val, proof_dic)
        value = not value
        sub_node_list.append(sub_node)
        f0case=1
    elif f[0] in ['and', '&', '&&']:
        sub_node_mode='and'
        cnt=1
        value = True
        for i in range(1,len(f)):
            value,sub_node=checkLTL(f[i], f_wait, t, trace, loop_start, vocab, c, v, formula_start+cnt, expect_val, proof_dic)
            cnt+=formula_len(f[i],len_cache) 
            sub_node_list.append(sub_node)
            if value is False:
                break
        f0case=2

    elif f[0] in ['or', '||','|']:
        cnt = 1
        value = False
        for i in range(1,len(f)):
            value,sub_node=checkLTL(f[i], f_wait, t, trace, loop_start, vocab, c, v, formula_start+cnt, expect_val, proof_dic)
            cnt+=formula_len(f[i],len_cache) 
            sub_node_list.append(sub_node)
            if value is True:
                break
        f0case=3

    elif f[0] in ['imp', '->']:

        value, sub_node = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v, formula_start + 1, 1 - expect_val, proof_dic)
        value = not value
        sub_node_list.append(sub_node)
        if value is False: 
            value, sub_node = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v, formula_start+1+formula_len(f[1],len_cache), proof_dic)
            sub_node_list.append(sub_node)
        switch_cnt[4] += 1

    elif f_wait[key]>1:
        # Confirm what your interpretation for this should be.
        if f[0] in ['G', 'F']:
            value, sub_node = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v, formula_start+1, expect_val, proof_dic)  # Confirm what your interpretation here should be
            sub_node_list.append(sub_node)
            f0case=5
            if f[0]=='F':
                f0case = 6
        elif f[0] == 'U':
            value, sub_node = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v, formula_start+1+formula_len(f[1],len_cache), expect_val, proof_dic)
            sub_node_list.append(sub_node)
            f0case=7
        elif f[0] == 'W':  # weak-until
            value, sub_node = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v, formula_start + 1, 1 - expect_val, proof_dic)
            sub_node_list.append(sub_node)
            if value is False:
                value, sub_node = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v,
                                             formula_start + 1 + formula_len(f[1], len_cache), proof_dic)
                sub_node_list.append(sub_node)

        elif f[0] == 'R':  # release (weak by default)

            value, sub_node = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v,
                                         formula_start + 1 + formula_len(f[1],len_cache), expect_val, proof_dic)
            sub_node_list.append(sub_node)
            f0case=8
            # value = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v)
        elif f[0] == 'X':
            value, sub_node = checkLTL(f[1], f_wait, t+1, trace, loop_start, vocab, c, v, formula_start + 1, expect_val, proof_dic)
            sub_node_list.append(sub_node)
            f0case=9

        else:
            # Does not exist in vocab, nor any of operators
            print(f,t,vocab)
            sys.exit('LTL check - something wrong')

    else:
        # Forward progression rules
        if f[0] == 'X':
            # value = checkLTL(f[1], f_wait, t + 1, trace, loop_start, vocab, c, v)
            value, sub_node = checkLTL(f[1], f_wait, t + 1, trace, loop_start, vocab, c, v, formula_start + 1,
                                       expect_val, proof_dic)
            sub_node_list.append(sub_node)
            f0case=10
        elif f[0] == 'G':
            sub_node_mode = 'and'
            value, sub_node = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v, formula_start + 1,
                                       expect_val, proof_dic)
            sub_node_list.append(sub_node)
            if value is True:
                value, sub_node = checkLTL(f, f_wait, t + 1, trace, loop_start, vocab, c, v, formula_start, expect_val, proof_dic)
                sub_node_list.append(sub_node)
            f0case = 11
        elif f[0] == 'F':
            value, sub_node = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v, formula_start + 1,
                                       expect_val, proof_dic)
            sub_node_list.append(sub_node)
            if value is False:
                value, sub_node = checkLTL(f, f_wait, t + 1, trace, loop_start, vocab, c, v, formula_start, expect_val, proof_dic)
                sub_node_list.append(sub_node)
            f0case = 12
            # value = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v) or checkLTL(('F', f[1]), f_wait, t + 1, trace, loop_start, vocab, c, v)
        elif f[0] == 'U' or f[0]=='W':
            # Basically enforces f[1] has to occur for f[1] U f[2] to be valid.
            value, sub_node = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v,
                                         formula_start + 1 + formula_len(f[1],len_cache), expect_val, proof_dic)
            sub_node_list.append(sub_node)
            if value is False:
                if expect_val == True:
                    sub_node_list=[]
                    sub_node_mode='and'
                value, sub_node = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v,
                                           formula_start + 1, expect_val, proof_dic)
                sub_node_list.append(sub_node)
                if value is True:
                    value, sub_node = checkLTL(f, f_wait, t+1, trace, loop_start, vocab, c, v,
                                               formula_start, expect_val, proof_dic)
                    if expect_val == False:
                        sub_node_list[-1]=sub_node
                    else:
                        sub_node_list.append(sub_node)

            f0case = 13

        elif f[0] == 'R':  # release (weak by default)

            sub_node_mode = 'and'
            value, sub_node = checkLTL(f[2], f_wait, t, trace, loop_start, vocab, c, v,
                                         formula_start + 1 + formula_len(f[1],len_cache), expect_val, proof_dic)
            sub_node_list.append(sub_node)
            f0case = 14
            if value == True:
                if expect_val == False:
                    sub_node_list=[]
                    sub_node_mode='or'
                value, sub_node = checkLTL(f[1], f_wait, t, trace, loop_start, vocab, c, v,
                                           formula_start + 1, expect_val, proof_dic)
                sub_node_list.append(sub_node)
                if value == False:
                    value, sub_node = checkLTL(f, f_wait, t+1, trace, loop_start, vocab, c, v,
                                               formula_start, expect_val, proof_dic)
                    if expect_val == True:
                        sub_node_list[-1]=sub_node
                    else:
                        sub_node_list.append(sub_node)

        else:
            # Does not exist in vocab, nor any of operators
            print(f, t, vocab)
            sys.exit('LTL check - something wrong')

    if v: print('Returned: ' + str(value))

    if expect_val == value:
        if expect_val == False:
            if sub_node_mode == 'and':
                sub_node_list = sub_node_list[-1:]
        if expect_val == True:
            if sub_node_mode == 'or':
                sub_node_list = sub_node_list[-1:]

        cof=len(sub_node_list)
        switch_cnt[cof*15-15+f0case]+=1
        # if f[0] == 'U':
        #     print('pre-curnode:', proof_node, sub_node_list)
        if proof_dic.get(proof_node,-1)!=-1:
            a=set(sub_node_list)
            a=a.union(proof_dic[proof_node])
            proof_dic[proof_node]=a
        else:
            proof_dic[proof_node]=set(sub_node_list)
        if proof_node in proof_dic[proof_node]:
            proof_dic[proof_node].remove(proof_node)
        # if f[0] == 'U':
        #     print('after-curnode:', proof_node, proof_dic[proof_node])
    # Save result
    if c is not None and type(c) is dict:
        key = (f, t, trace['name'])
        c[key] = value  # append

    if printproof:
        print('state:'+str(t) + ':' + str(trace['trace'][t]) + ', sub formula:' + str(f) + 'is '+ str(value))
    # proofcnt -= 1
    return (value,proof_node)


def preorder_turple(f):
    if isinstance(f, LTLfAtomic):
        return f.s.replace('true', '1').replace('false', '0')
    if isinstance(f, LTLfAnd) or isinstance(f, LTLfUntil) or isinstance(f, LTLfOr) or isinstance(f, LTLfRelease) or isinstance(f, LTLfImplies):
        if len(f.formulas)>2:
            nf=deepcopy(f)
            nf.formulas=nf.formulas[1:]
            return (f.operator_symbol,preorder_turple(f.formulas[0]),preorder_turple(nf))
        return tuple([f.operator_symbol.replace('R', 'W')]+[preorder_turple(f.formulas[i]) for i in range(len(f.formulas))])
    if isinstance(f, LTLfNot) or isinstance(f, LTLfNext) or isinstance(f, LTLfAlways) or isinstance(f, LTLfEventually):
        return (f.operator_symbol,preorder_turple(f.f))

def ltl2turple(ltl: str):
    parser = LTLfParser()
    formula = parser(ltl)
    return preorder_turple(formula)


def check(ltl,trace,vocab):
    t,loop_start=trace
    trace={'name':'t','trace':t}
    formula=ltl2turple(ltl)
    if printproof:
        print('formula',formula)
        print('trace',t)
        print('loop_start',loop_start)
    proof_dic={}
    value,root_node=checkLTL(formula, {}, 0, trace, loop_start, vocab, {}, False, 0, 1, proof_dic)

    return value

if __name__ == '__main__':

    ltl='p0 U p1'
    trace=([['p0'], ['p3','p2']],-1)
    vocab={f'p{i}' for i in range(10)}
    value=check(ltl,trace,vocab)
    print('value',value)