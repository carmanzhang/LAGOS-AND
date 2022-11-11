import collections
import os

import numpy as np
from matplotlib import pyplot as plot

from myconfig import cached_dir, latex_doc_base_dir

plot.rcParams['font.family'] = 'serif'
plot.rcParams['font.serif'] = ['Times New Roman'] + plot.rcParams['font.serif']

from mytookit.data_reader import DBReader

colors = ['green', 'gold', 'red', 'black', 'cyan', 'blue', 'magenta', 'purple', 'gray', 'fuchsia', 'orange', 'yellow']
linestyles = ['--', '-.', ':', '--']
line_markers = ['<', '>', '^', 'v']
linewidth = 4
tick_font_size = 14
df_whole = DBReader.tcp_model_cached_read(os.path.join(cached_dir, "whole_mag_representativeness_distribution.pkl"),
                                          "select * from and_ds.whole_mag_representativeness_distribution;",
                                          cached=True)
print('df_whole.shape', df_whole.shape)
# ['check_item' 'distribution']
print(df_whole['check_item'].values)
# for i, (check_item, distribution) in df_whole.iterrows():
#     print(check_item, len(distribution), distribution[:5])
# pub_year, author_position, lastname_popularity, ssn_gender-sex, sex_mac-sex, ethnic-seer, genni-sex, ethnea
whole_pub_year_dist = df_whole[df_whole['check_item'] == 'pub_year']['distribution'].values[0]
whole_author_position_dist = df_whole[df_whole['check_item'] == 'author_position']['distribution'].values[0]
whole_mac_gender_dist = df_whole[df_whole['check_item'] == 'sex_mac-sex']['distribution'].values[0]
whole_genni_gender_dist = df_whole[df_whole['check_item'] == 'genni-sex']['distribution'].values[0]
whole_ssn_gender_dist = df_whole[df_whole['check_item'] == 'ssn_gender-sex']['distribution'].values[0]
whole_lastname_popularity_dist = df_whole[df_whole['check_item'] == 'lastname_popularity']['distribution'].values[0]
whole_lastname_first_initial_popularity_dist = \
    df_whole[df_whole['check_item'] == 'lastname_first_initial_popularity']['distribution'].values[0]
# whole_lastname_popularity_dist = df_whole[df_whole['check_item'] == 'lastname']['distribution'].values[0]
# whole_lastname_first_initial_popularity_dist = df_whole[df_whole['check_item'] == 'lastname_first_initial']['distribution'].values[0]
whole_ethnic_seer_dist = df_whole[df_whole['check_item'] == 'ethnic-seer']['distribution'].values[0]
whole_ethnea_dist = df_whole[df_whole['check_item'] == 'ethnea']['distribution'].values[0]
whole_fos_dist = df_whole[df_whole['check_item'] == 'fos']['distribution'].values[0]

print(whole_pub_year_dist[:5])
print(whole_author_position_dist[:5])
print(whole_mac_gender_dist[:5])
print(whole_genni_gender_dist[:5])
print(whole_ssn_gender_dist[:5])
print(whole_lastname_popularity_dist[:5])
print(whole_ethnic_seer_dist[:5])
print(whole_ethnea_dist[:5])

df_block = DBReader.tcp_model_cached_read(os.path.join(cached_dir, "orcid_mag_matched_representativeness.pkl"),
                                          sql="""select * from and_ds.our_dataset_representativeness;""",
                                          cached=True)
# ['pid' 'orcid' 'author_position' 'lastname' 'ethnic_seer' 'ethnea' 'genni', 'sex_mac' 'ssn_gender' 'pub_year']
print(len(df_block.columns.values), df_block.columns.values)
# df_sample = df_block[:10]

# Note distribution of various aspects of the block-based dataset
pub_year_counter_block = sorted(collections.Counter(df_block['pub_year'].values).items(), key=lambda x: x[0], reverse=False)
author_position_counter_block = sorted(collections.Counter(df_block['author_position'].values).items(), key=lambda x: x[0],
                                       reverse=False)
author_genni_gender_counter_block = sorted(collections.Counter(df_block['genni'].values).items(), key=lambda x: x[0],
                                           reverse=False)

author_sex_mac_counter_block = sorted(collections.Counter(df_block['sex_mac'].values).items(), key=lambda x: x[0],
                                      reverse=False)
author_ssn_gender_counter_block = sorted(collections.Counter(df_block['ssn_gender'].values).items(), key=lambda x: x[0],
                                         reverse=False)

author_ethnic_seer_counter_block = sorted(collections.Counter(df_block['ethnic_seer'].values).items(), key=lambda x: x[0],
                                          reverse=False)
author_ethnea_counter_block = sorted(collections.Counter(df_block['ethnea'].values).items(), key=lambda x: x[0],
                                     reverse=False)
author_lastname_counter_block = sorted(collections.Counter(df_block['lastname'].values).items(), key=lambda x: x[1],
                                       reverse=True)
author_lastname_counter_block = sorted(collections.Counter([n[1] for n in author_lastname_counter_block]).items(),
                                       key=lambda x: x[0],
                                       reverse=True)
lastname_first_initial_counter_block = sorted(collections.Counter(df_block['lastname_first_initial'].values).items(),
                                              key=lambda x: x[1], reverse=True)
lastname_first_initial_counter_block = sorted(collections.Counter([n[1] for n in lastname_first_initial_counter_block]).items(),
                                              key=lambda x: x[0],
                                              reverse=True)
fos_counter_block = sorted(
    collections.Counter([n for n in np.hstack(df_block['fos_arr'].values) if len(n) > 0]).items(), key=lambda x: x[1],
    reverse=False)

# Note distribution of various aspects of the pairwise-based dataset
df_pairwise = DBReader.tcp_model_cached_read("xxx",
                                             sql="""select * from and_ds.our_dataset_pairwise_representativeness;""",
                                             cached=False)
print('df_pairwise.shape before adjustment', df_pairwise.shape)

pub_year_counter_pairwise = sorted(collections.Counter(df_pairwise['pub_year'].values).items(), key=lambda x: x[0], reverse=False)
author_position_counter_pairwise = sorted(collections.Counter(df_pairwise['author_position'].values).items(), key=lambda x: x[0],
                                          reverse=False)
author_genni_gender_counter_pairwise = sorted(collections.Counter(df_pairwise['genni'].values).items(), key=lambda x: x[0],
                                              reverse=False)
author_sex_mac_counter_pairwise = sorted(collections.Counter(df_pairwise['sex_mac'].values).items(), key=lambda x: x[0],
                                         reverse=False)
author_ssn_gender_counter_pairwise = sorted(collections.Counter(df_pairwise['ssn_gender'].values).items(), key=lambda x: x[0],
                                            reverse=False)
author_ethnic_seer_counter_pairwise = sorted(collections.Counter(df_pairwise['ethnic_seer'].values).items(), key=lambda x: x[0],
                                             reverse=False)
author_ethnea_counter_pairwise = sorted(collections.Counter(df_pairwise['ethnea'].values).items(), key=lambda x: x[0],
                                        reverse=False)
author_lastname_counter_pairwise = sorted(collections.Counter(df_pairwise['lastname'].values).items(), key=lambda x: x[1],
                                          reverse=True)
author_lastname_counter_pairwise = sorted(collections.Counter([n[1] for n in author_lastname_counter_pairwise]).items(),
                                          key=lambda x: x[0],
                                          reverse=True)
lastname_first_initial_counter_pairwise = sorted(collections.Counter(df_pairwise['lastname_first_initial'].values).items(),
                                                 key=lambda x: x[1], reverse=True)
lastname_first_initial_counter_pairwise = sorted(
    collections.Counter([n[1] for n in lastname_first_initial_counter_pairwise]).items(),
    key=lambda x: x[0],
    reverse=True)
fos_counter_pairwise = sorted(
    collections.Counter([n for n in np.hstack(df_pairwise['fos_arr'].values) if len(n) > 0]).items(), key=lambda x: x[1],
    reverse=False)


def plot_pub_year(whole_pub_year_dist, counter_block, counter_pairwise, check_item):
    whole_pub_year_dist = [n for n in whole_pub_year_dist if 1970 <= int(n[0]) <= 2018]
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [int(n[0]) for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter1 = [n for n in counter_block if 1970 <= n[0] <= 2018]
    pub_cnt1 = sum([n[1] for n in pub_year_counter1])
    pub_year1 = [n[0] for n in pub_year_counter1]
    pub_count1 = [n[1] * 1.0 / pub_cnt1 for n in pub_year_counter1]

    pub_year_counter2 = [n for n in counter_pairwise if 1970 <= n[0] <= 2018]
    pub_cnt2 = sum([n[1] for n in pub_year_counter2])
    pub_year2 = [n[0] for n in pub_year_counter2]
    pub_count2 = [n[1] * 1.0 / pub_cnt2 for n in pub_year_counter2]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year1, pub_count1, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-BLOCK', linewidth=linewidth)
    idx = 2
    plot.plot(pub_year2, pub_count2, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-PAIRWISE', linewidth=linewidth)

    # plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('Year', loc='right', fontsize=18)
    plot.ylabel('Proportion', loc='center', fontsize=18)  # 'top'
    plot.xticks(fontsize=tick_font_size)
    plot.yticks(fontsize=tick_font_size)
    plot.legend(loc='best')  # 'lower right'


def plot_author_position(whole_author_position_dist, counter_block, counter_pairwise, check_item):
    whole_pub_year_dist = [n for n in whole_author_position_dist if int(n[0]) <= 15]
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [int(n[0]) for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter1 = [n for n in counter_block if n[0] <= 15]
    pub_cnt1 = sum([n[1] for n in pub_year_counter1])
    pub_year1 = [n[0] for n in pub_year_counter1]
    pub_count1 = [n[1] * 1.0 / pub_cnt1 for n in pub_year_counter1]

    pub_year_counter2 = [n for n in counter_pairwise if n[0] <= 15]
    pub_cnt2 = sum([n[1] for n in pub_year_counter2])
    pub_year2 = [n[0] for n in pub_year_counter2]
    pub_count2 = [n[1] * 1.0 / pub_cnt2 for n in pub_year_counter2]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year1, pub_count1, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-BLOCK', linewidth=linewidth)
    idx = 2
    plot.plot(pub_year2, pub_count2, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-PAIRWISE', linewidth=linewidth)

    plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('Author Position', loc='right', fontsize=18)
    plot.ylabel('Proportion', loc='center', fontsize=18)  # 'top'
    plot.xticks(fontsize=tick_font_size)
    plot.yticks(fontsize=tick_font_size)
    plot.legend(loc='best')  # 'lower right'


def plot_author_gender(whole_genni_gender_dist, counter_block, counter_pairwise, check_item):
    x_label_map = {'-': 'Unsure', 'F': 'Female', 'M': 'Male', '': ''}
    whole_pub_year_dist = sorted(whole_genni_gender_dist, key=lambda x: x[0], reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [x_label_map[n[0]] for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter1 = sorted([n for n in counter_block if x_label_map[n[0]] in set(whole_pub_year)],
                               key=lambda x: x[0], reverse=False)
    pub_cnt1 = sum([n[1] for n in pub_year_counter1])
    pub_year1 = [x_label_map[n[0]] for n in pub_year_counter1]
    pub_count1 = [n[1] * 1.0 / pub_cnt1 for n in pub_year_counter1]

    pub_year_counter2 = sorted([n for n in counter_pairwise if x_label_map[n[0]] in set(whole_pub_year)],
                               key=lambda x: x[0], reverse=False)
    pub_cnt2 = sum([n[1] for n in pub_year_counter2])
    pub_year2 = [x_label_map[n[0]] for n in pub_year_counter2]
    pub_count2 = [n[1] * 1.0 / pub_cnt2 for n in pub_year_counter2]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year1, pub_count1, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-BLOCK', linewidth=linewidth)
    idx = 2
    plot.plot(pub_year2, pub_count2, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-PAIRWISE', linewidth=linewidth)

    plot.title(check_item, fontsize=18)
    plot.xlabel('Gender', loc='right', fontsize=18)
    plot.ylabel('Proportion', loc='center', fontsize=18)  # 'top'
    plot.xticks(fontsize=tick_font_size)
    plot.yticks(fontsize=tick_font_size)

    plot.legend(loc='best')  # 'lower right'


def plot_ethnic_seer(whole_ethnic_seer_dist, counter_block, counter_pairwise, check_item):
    whole_pub_year_dist = sorted(whole_ethnic_seer_dist, key=lambda x: x[1], reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [n[0] for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    keys1 = [n[0] for n in counter_block]
    pub_year_counter1 = [counter_block[keys1.index(n)] if n in keys1 else (n, 0) for n in whole_pub_year]
    pub_cnt1 = sum([n[1] for n in pub_year_counter1])
    pub_year1 = [n[0] for n in pub_year_counter1]
    pub_count1 = [n[1] * 1.0 / pub_cnt1 for n in pub_year_counter1]

    keys2 = [n[0] for n in counter_pairwise]
    pub_year_counter2 = [counter_block[keys2.index(n)] if n in keys2 else (n, 0) for n in whole_pub_year]
    pub_cnt2 = sum([n[1] for n in pub_year_counter2])
    pub_year2 = [n[0] for n in pub_year_counter2]
    pub_count2 = [n[1] * 1.0 / pub_cnt2 for n in pub_year_counter2]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year1, pub_count1, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-BLOCK', linewidth=linewidth)
    idx = 2
    plot.plot(pub_year2, pub_count2, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-PAIRWISE', linewidth=linewidth)

    # plot.xscale('log')
    plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('Ethnicity', loc='right', fontsize=18)
    plot.ylabel('Proportion', loc='center', fontsize=18)  # 'top'
    plot.xticks(fontsize=tick_font_size - 4)
    plot.yticks(fontsize=tick_font_size)

    plot.legend(loc='best')  # 'lower right'


# def plot_ethnea(whole_ethnic_seer_dist, author_ethnic_seer_counter, check_item):
#     whole_pub_year_dist = sorted(whole_ethnic_seer_dist, key=lambda x: x[1], reverse=False)
#     all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
#     whole_pub_year = [n[0] for n in whole_pub_year_dist]
#     whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]
#
#     keys = [n[0] for n in author_ethnic_seer_counter]
#     pub_year_counter = [author_ethnic_seer_counter[keys.index(n)] if n in keys else (n, 0) for n in whole_pub_year]
#     pub_cnt = sum([n[1] for n in pub_year_counter])
#     pub_year = [n[0] for n in pub_year_counter]
#     pub_count = [n[1] * 1.0 / pub_cnt for n in pub_year_counter]
#
#     # plot.figure()
#     idx = 0
#     plot.loglog(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
#                 # marker=line_markers[idx], markersize=8, markevery=0.2,
#                 color=colors[idx], label='MAG', linewidth=linewidth)
#     idx = 1
#     plot.loglog(pub_year, pub_count, linestyle=linestyles[idx],
#                 # marker=line_markers[idx], markersize=8, markevery=0.2,
#                 color=colors[idx], label='LAGOS-AND', linewidth=linewidth)
#     plot.title(check_item, fontsize=18)
#     plot.xlabel('ethnicity', loc='right', fontsize=18)
#     plot.ylabel('ethnicity proportion', loc='center', fontsize=18)  # 'top'
#     plot.legend(loc='best')  # 'lower right'


def plot_lastname_popularity(whole_lastname_popularity_dist, counter_block, counter_pairwise, check_item):
    # ratio = len(author_lastname_counter) * 1.0 / len(whole_lastname_popularity_dist)
    # used_for_plot_ratio = 1
    whole_pub_year_dist = sorted([n for n in whole_lastname_popularity_dist],
                                 # if random() <= used_for_plot_ratio * ratio
                                 key=lambda x: int(x[0]), reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [int(n[0]) for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter1 = sorted([n for n in counter_block],  # if random() <= used_for_plot_ratio
                               key=lambda x: int(x[0]), reverse=False)
    pub_cnt1 = sum([n[1] for n in pub_year_counter1])
    pub_year1 = [int(n[0]) for n in pub_year_counter1]
    pub_count1 = [n[1] * 1.0 / pub_cnt1 for n in pub_year_counter1]

    pub_year_counter2 = sorted([n for n in counter_pairwise],  # if random() <= used_for_plot_ratio
                               key=lambda x: int(x[0]), reverse=False)
    pub_cnt2 = sum([n[1] for n in pub_year_counter2])
    pub_year2 = [int(n[0]) for n in pub_year_counter2]
    pub_count2 = [n[1] * 1.0 / pub_cnt2 for n in pub_year_counter2]

    print(whole_pub_year_dist[0], whole_pub_year_dist[-1])
    print(pub_year_counter1[0], pub_year_counter1[-1])
    print(pub_year_counter2[0], pub_year_counter2[-1])

    print(list(zip(whole_pub_year, whole_pub_dist)))
    print(list(zip(pub_year1, pub_count1)))
    print(list(zip(pub_year2, pub_count2)))

    # plot.figure()
    idx = 0
    plot.scatter(whole_pub_year,  # [n * 100.0 / len(whole_pub_dist) for n in range(len(whole_pub_dist))],
                 whole_pub_dist,
                 marker='.',
                 color=colors[idx], label='MAG', s=4)
    idx = 1
    plot.scatter(pub_year1,  # [n * 100.0 / len(pub_year) for n in range(len(pub_year))],
                 pub_count1,
                 marker='o',
                 color=colors[idx], label='LAGOS-AND-BLOCK', s=4)
    idx = 2
    plot.scatter(pub_year2,  # [n * 100.0 / len(pub_year) for n in range(len(pub_year))],
                 pub_count2,
                 marker='s',
                 color=colors[idx], label='LAGOS-AND-PAIRWISE', s=4)

    plot.xscale('log')
    plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('LN Popularity', loc='right', fontsize=18)
    plot.ylabel('Proportion', loc='center', fontsize=18)  # 'top'
    plot.xticks(fontsize=tick_font_size)
    plot.yticks(fontsize=tick_font_size)

    plot.legend(loc='best')  # 'lower right'


def plot_namespace_popularity(whole_lastname_popularity_dist, counter_block, counter_pairwise, check_item):
    # ratio = len(author_lastname_counter) * 1.0 / len(whole_lastname_popularity_dist)
    # used_for_plot_ratio = 1
    whole_pub_year_dist = sorted([n for n in whole_lastname_popularity_dist],
                                 # if random() <= used_for_plot_ratio * ratio
                                 key=lambda x: int(x[0]), reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [int(n[0]) for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter1 = sorted([n for n in counter_block],  # if random() <= used_for_plot_ratio
                               key=lambda x: int(x[0]), reverse=False)
    pub_cnt1 = sum([n[1] for n in pub_year_counter1])
    pub_year1 = [int(n[0]) for n in pub_year_counter1]
    pub_count1 = [n[1] * 1.0 / pub_cnt1 for n in pub_year_counter1]

    pub_year_counter2 = sorted([n for n in counter_pairwise],  # if random() <= used_for_plot_ratio
                               key=lambda x: int(x[0]), reverse=False)
    pub_cnt2 = sum([n[1] for n in pub_year_counter2])
    pub_year2 = [int(n[0]) for n in pub_year_counter2]
    pub_count2 = [n[1] * 1.0 / pub_cnt2 for n in pub_year_counter2]

    print(whole_pub_year_dist[0], whole_pub_year_dist[-1])
    print(pub_year_counter1[0], pub_year_counter1[-1])
    print(pub_year_counter2[0], pub_year_counter2[-1])

    print(list(zip(whole_pub_year, whole_pub_dist)))
    print(list(zip(pub_year1, pub_count1)))
    print(list(zip(pub_year2, pub_count2)))

    # plot.figure()
    idx = 0
    plot.scatter(whole_pub_year,  # [n * 100.0 / len(whole_pub_dist) for n in range(len(whole_pub_dist))],
                 whole_pub_dist,
                 marker='.',
                 color=colors[idx], label='MAG', s=4)
    idx = 1
    plot.scatter(pub_year1,  # [n * 100.0 / len(pub_year) for n in range(len(pub_year))],
                 pub_count1,
                 marker='o',
                 color=colors[idx], label='LAGOS-AND-BLOCK', s=4)
    idx = 2
    plot.scatter(pub_year2,  # [n * 100.0 / len(pub_year) for n in range(len(pub_year))],
                 pub_count2,
                 marker='s',
                 color=colors[idx], label='LAGOS-AND-PAIRWISE', s=4)

    plot.xscale('log')
    plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('LNFI Popularity', loc='right', fontsize=18)
    plot.ylabel('Proportion', loc='center', fontsize=18)  # 'top'
    plot.xticks(fontsize=tick_font_size)
    plot.yticks(fontsize=tick_font_size)

    plot.legend(loc='best')  # 'lower right'


def plot_fos(whole_fos_dist, counter_block, counter_pairwise, check_item):
    whole_pub_year_dist = sorted(whole_fos_dist, key=lambda x: x[1], reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [n[0] for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    keys1 = [n[0] for n in counter_block]
    print(keys1)
    pub_year_counter1 = [counter_block[keys1.index(n)] if n in keys1 else (n, 0) for n in whole_pub_year]
    pub_cnt1 = sum([n[1] for n in pub_year_counter1])
    pub_year1 = [n[0] for n in pub_year_counter1]
    pub_count1 = [n[1] * 1.0 / pub_cnt1 for n in pub_year_counter1]

    keys2 = [n[0] for n in counter_pairwise]
    print(keys2)
    pub_year_counter2 = [counter_block[keys2.index(n)] if n in keys2 else (n, 0) for n in whole_pub_year]
    pub_cnt2 = sum([n[1] for n in pub_year_counter2])
    pub_year2 = [n[0] for n in pub_year_counter2]
    pub_count2 = [n[1] * 1.0 / pub_cnt2 for n in pub_year_counter2]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year1, pub_count1, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-BLOCK', linewidth=linewidth)
    idx = 2
    plot.plot(pub_year2, pub_count2, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND-PAIRWISE', linewidth=linewidth)
    # plot.yscale('log')
    plot.xticks(fontsize=10, rotation=45, ha='right')
    # plot.autofmt_xdate(bottom=0.2, rotation=30, ha='center')
    plot.title(check_item, fontsize=18)
    # plot.xlabel('domain', loc='right')
    plot.ylabel('Proportion', loc='center', fontsize=18)  # 'top'
    # plot.xticks(fontsize=tick_font_size)
    plot.yticks(fontsize=tick_font_size)

    plot.legend(loc='best')  # 'lower right'


plot.figure(42, figsize=(12, 18), dpi=300)
plot.subplot(421)
# plot.grid(True)
plot_pub_year(whole_pub_year_dist, pub_year_counter_block, pub_year_counter_pairwise,
              check_item='(a) Publication Distribution')
plot.subplot(422)
plot_author_position(whole_author_position_dist, author_position_counter_block, author_position_counter_pairwise,
                     check_item='(b) Author Position Distribution')
plot.subplot(423)

plot_author_gender(whole_genni_gender_dist, author_genni_gender_counter_block, author_genni_gender_counter_pairwise,
                   check_item='(c) Gender Distribution')
plot.subplot(424)
# plot_author_gender(whole_mac_gender_dist, author_sex_mac_counter, check_item='mac_gender')
# plot_author_gender(whole_ssn_gender_dist, author_ssn_gender_counter, check_item='ssn_gender')

plot_ethnic_seer(whole_ethnic_seer_dist, author_ethnic_seer_counter_block, author_ethnic_seer_counter_pairwise,
                 check_item='(d) Ethnicity Distribution')

# plot_ethnea(whole_ethnea_dist, author_ethnea_counter, check_item='(d) ethnicity distribution')
# plot.subplot(425)

plot.subplot(425)
plot_lastname_popularity(whole_lastname_popularity_dist, author_lastname_counter_block, author_lastname_counter_pairwise,
                         check_item='(e) LN Popularity Distribution')
plot.subplot(426)
plot_namespace_popularity(whole_lastname_first_initial_popularity_dist, lastname_first_initial_counter_block,
                          lastname_first_initial_counter_pairwise,
                          check_item='(f) LNFI Popularity Distribution')

plot.subplot(427)
plot_fos(whole_fos_dist, fos_counter_block, fos_counter_pairwise, check_item='(g) Domain Distribution')

plot.tight_layout()
plot.savefig(os.path.join(cached_dir, 'data-distribution.png'), dpi=600)
plot.savefig(os.path.join(latex_doc_base_dir, 'figs/data-distribution.png'), dpi=600)
# plot.savefig(os.path.join(cached_dir, 'gold-standard-check.pdf'), dpi=500)
plot.show()
