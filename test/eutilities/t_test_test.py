# # Calculate the T-test for the means of *two independent* samples of scores.
# stats.ttest_ind()
#
# # Calculate the t-test on TWO RELATED samples of scores, a and b.
# stats.ttest_rel()
#
# # Calculate the T-test for the mean of ONE group of scores.
# stats.ttest_1samp(paired,0)

# from scipy import stats
# #单一样本的t检验，检验单一样本是否与给定的均值popmean差异显著的函数，第一个参数为给定的样本，第二个函数为给定的均值popmean，可以以列表的形式传输多个单一样本和均值。
# stats.ttest_1samp(a, popmean, axis=0, nan_policy='propagate')
# #独立样本的T检验，检验两个样本的均值差异，该检验方法假定了样本的通过了F检验，即两个独立样本的方差相同
# stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')
# #检验两个样本的均值差异（同上），输出的参数两个样本的统计量，包括均值，标准差，和样本大小
# stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, equal_var=True)
# #配对T检验，检测两个样本的均值差异，输入的参数是样本的向量
# stats.ttest_rel(a, b, axis=0, nan_policy='propagate')

import numpy as np
from scipy import stats

np.random.seed(12345678)  # fix random seed to get same numbers

rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
rvs2 = (stats.norm.rvs(loc=5, scale=10, size=500) + stats.norm.rvs(scale=0.2, size=500))
stats.ttest_rel(rvs1, rvs2)
rvs3 = (stats.norm.rvs(loc=8, scale=10, size=500) +
        stats.norm.rvs(scale=0.2, size=500))
rel = stats.ttest_rel(rvs1, rvs3)
# print(rel.index(1))
print(rel.pvalue, rel.statistic)
