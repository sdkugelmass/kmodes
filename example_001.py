import pandas as pd
import numpy as np
import time

from kmodes.kprototypes_fast import KPrototypes as KPrototypesFast
from kmodes.kprototypes import KPrototypes as KPrototypesReg


# from kmodes.util.dissim_cy import matching_dissim_int_cy
from kmodes.util.dissim import matching_dissim

df_emp = pd.read_pickle(
    '/home/sdk/Work/Driscoll/src/RISK_AI_2021/data_2022-01-15/df_emp.pkl')
df_emp = df_emp.sample(n=100000)
dfMatrix = df_emp.to_numpy()

cost = []
n_cluster = 5

kproto_reg = KPrototypesReg(cat_dissim=matching_dissim, n_init=10,
                            n_jobs=-1, n_clusters=n_cluster, init='Huang',
                            random_state=42, verbose=0)

kproto_fast = KPrototypesFast(cat_dissim=matching_dissim, n_init=10,
                              n_jobs=-1, n_clusters=n_cluster, init='Huang',
                              random_state=42, verbose=1)


# time_reg_start = time.time()
# kproto_reg.fit(df_emp, categorical=[0, 1, 2, 3])
# time_reg_stop = time.time()
# print(time_reg_stop - time_reg_start)

time_fast_start = time.time()
kproto_fast.fit(df_emp, categorical=[0, 1, 2, 3])
time_fast_stop = time.time()
print(time_fast_stop - time_fast_start)




# %prun kproto2.fit(df_emp, categorical=[0,1,2,3])


# kproto1 = KPrototypes(cat_dissim=matching_dissim_int_cy,
#                      n_jobs=1, n_clusters=n_cluster, init='Huang',
#                      random_state=42, verbose=1)


# result1 = kproto1.fit_predict2(df_emp, categorical=[0,1,2,3])

# time1 = time.time()
# result1 = kproto1.fit_predict2(dfMatrix, categorical=[0,1,2,3])
# tt1 = time.time() - time1


# time1 = time.time()
# result2 = kproto2.fit_predict2(dfMatrix, categorical=[0,1,2,3])
# tt2 = time.time() - time1

# %prun kproto1.fit_predict2(dfMatrix, categorical=[0,1,2,3])
# %prun kproto2.fit_predict2(dfMatrix, categorical=[0,1,2,3])

# a = np.random.randint(0, 10, (1_000_000,4)).astype(np.uint32)
# b = np.random.randint(0, 10, (4,)).astype(np.uint32)
