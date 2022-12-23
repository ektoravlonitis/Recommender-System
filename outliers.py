# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:29:37 2022

@author: mypc
"""

import numpy as np
from scipy import stats
#takes x as entire df, thresh as threshold (should be 0-1 ??), cl as a dataframe with 1 column
#function returns the df without any rows who have a z value below the threshold
#returns a outlier-free DF
def outexter(x,thresh,cl):
    if type(cl)==str:
        j=type(x[cl][1])
        x=x[x[cl].apply(type)==j]
        tested=x[cl].apply(np.float64)
        z = np.abs(stats.zscore(tested))
        x['z']=z
        l=(x[x['z']<thresh])
        l=l.drop('z',axis=1)
    else:
        for i in cl:
            l=outexter(x,thresh,i)
    return l
