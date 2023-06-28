import pandas as pd
import numpy as np
import threading

df = pd.read_csv('../datasets/raw_brands.csv', sep=';')
df.drop_duplicates(subset='name', inplace=True)

def text_processing(string):
    var = ''.join(x for x in string if x.isalpha() or x == ' ')
    return var.strip()

df['edited'] = df['name'].str.lower()
df['edited'] = df['edited'].apply(text_processing)
df.drop(df[df['edited'].map(len) == 0].index, inplace=True)
df.reset_index(drop=True, inplace=True)
df['closest'] = None

def matches_transpose(st1: str, st2: str):
    mx = st1 if len(st1) > len(st2) else st2
    mn = st2 if len(st2) < len(st1) else st1 if mx != st1 else st2
    match = 0
    trans = 0
    for i in range(len(mn)):
        for j in range(len(mx)):
            if mn[i] == mx[j]:
                match += 1
                if i != j:
                    trans += 1
                break
    return match, trans

def jaro(st1: str, st2: str):
    l1 = len(st1)
    l2 = len(st2)

    match, transpos = matches_transpose(st1, st2)
    if match == 0:
        return 0
    return 1/3 * (match/l1 + match/l2 + (match - transpos)/match)

def jaro_winkler(st1: str, st2: str):
    jar = jaro(st1, st2)
    if jar == 0:
        return 0
    p = 0
    while p < len(min(st1, st2, key=len)) and st1[p] == st2[p] and p < 4:
        p += 1
    return jar + 0.1 * p * (1 - jar)

def comparison(left, right):
    size = df.shape[0]
    for i in range(left, right):
        lst = []
        for j in range(size):
            if i == j:
                continue
            s1 = df.loc[i, 'edited']
            s2 = df.loc[j, 'edited']
            if jaro_winkler(s1, s2) >= 0.8:
                lst.append(j)
        if len(lst) > 0:
            df['closest'][i] = lst


def split_to_threads():
    pool = int(np.linspace(0, df.shape[0], 7)[1])

    t1 = threading.Thread(target=comparison, args=(0, pool))
    t2 = threading.Thread(target=comparison, args=(pool, pool*2))
    t3 = threading.Thread(target=comparison, args=(pool*2, pool*3))
    t4 = threading.Thread(target=comparison, args=(pool*3, pool*4))
    t5 = threading.Thread(target=comparison, args=(pool*4, pool*5))
    t6 = threading.Thread(target=comparison, args=(pool*5, df.shape[0]))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()

split_to_threads()

df.drop(columns=['edited']).to_csv('../datasets/result.csv')
