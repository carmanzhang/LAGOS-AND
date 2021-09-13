import collections

import numpy as np
from matplotlib import pyplot as plot

plot.rcParams['font.family'] = 'serif'
plot.rcParams['font.serif'] = ['Times New Roman'] + plot.rcParams['font.serif']

from myio.data_reader import DBReader

colors = ['green', 'red', 'gold', 'black', 'cyan', 'blue', 'magenta', 'purple', 'gray', 'fuchsia', 'orange', 'yellow']
linestyles = ['--', '-.', '--', '--']
line_markers = ['<', '>', '^', 'v']
linewidth = 4
df_whole = DBReader.tcp_model_cached_read("../cached/whole_mag_representativeness_distribution.pkl",
                                          "select * from and_ds.materialized_whole_mag_representativeness_distribution;",
                                          cached=True)
print('df_whole.shape', df_whole.shape)
columns = df_whole.columns.values
# ['check_item' 'distribution']
print(len(columns), columns)
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

df_matched = DBReader.tcp_model_cached_read("../cached/orcid_mag_matched_representativeness.pkl", """
                            select pid,
                                   orcid,
                                   orcid_names,
                                   author_position,
                                   splitByChar(' ', matched_biblio_author)[-1] as lastname,
                                   num_iag_in_block,
                                   num_citation_in_block,
                                   num_citation_in_iag,
                                   varied_last_name,
                                   concat(lastname, '_', substring(matched_biblio_author, 1, 1)) as lastname_first_initial,
                                   ethnic_seer,
                                   ethnea,
                                   genni,
                                   sex_mac,
                                   ssn_gender,
                                   pub_year,
                                   fos_arr
                            from and_ds.our_dataset_representativeness;""",
                                            cached=True)
print('df_matched.shape before adjustment', df_matched.shape)

columns = df_matched.columns.values

# ['pid' 'orcid' 'author_position' 'lastname' 'ethnic_seer' 'ethnea' 'genni', 'sex_mac' 'ssn_gender' 'pub_year']
print(len(columns), columns)
df_sample = df_matched[:10]

pub_year_counter = sorted(collections.Counter(df_matched['pub_year'].values).items(), key=lambda x: x[0], reverse=False)
author_position_counter = sorted(collections.Counter(df_matched['author_position'].values).items(), key=lambda x: x[0],
                                 reverse=False)
author_genni_gender_counter = sorted(collections.Counter(df_matched['genni'].values).items(), key=lambda x: x[0],
                                     reverse=False)

author_sex_mac_counter = sorted(collections.Counter(df_matched['sex_mac'].values).items(), key=lambda x: x[0],
                                reverse=False)
author_ssn_gender_counter = sorted(collections.Counter(df_matched['ssn_gender'].values).items(), key=lambda x: x[0],
                                   reverse=False)

author_ethnic_seer_counter = sorted(collections.Counter(df_matched['ethnic_seer'].values).items(), key=lambda x: x[0],
                                    reverse=False)
author_ethnea_counter = sorted(collections.Counter(df_matched['ethnea'].values).items(), key=lambda x: x[0],
                               reverse=False)
author_lastname_counter = sorted(collections.Counter(df_matched['lastname'].values).items(), key=lambda x: x[1],
                                 reverse=True)
author_lastname_counter = sorted(collections.Counter([n[1] for n in author_lastname_counter]).items(),
                                 key=lambda x: x[0],
                                 reverse=True)
lastname_first_initial_counter = sorted(collections.Counter(df_matched['lastname_first_initial'].values).items(),
                                        key=lambda x: x[1], reverse=True)
lastname_first_initial_counter = sorted(collections.Counter([n[1] for n in lastname_first_initial_counter]).items(),
                                        key=lambda x: x[0],
                                        reverse=True)
fos_counter = sorted(
    collections.Counter([n for n in np.hstack(df_matched['fos_arr'].values) if len(n) > 0]).items(), key=lambda x: x[1],
    reverse=False)

orcid_counter = sorted(collections.Counter(df_matched['orcid'].values).items(), key=lambda x: x[1], reverse=True)
author_group_size_counter = sorted(collections.Counter([n[1] for n in orcid_counter]).items(), key=lambda x: x[1],
                                   reverse=True)
orcid_names_counter = sorted(collections.Counter(df_matched['orcid_names'].values).items(), key=lambda x: x[1],
                             reverse=True)
block_size_counter = sorted(collections.Counter([n[1] for n in orcid_names_counter]).items(), key=lambda x: x[1],
                            reverse=True)


def aggregate_by_key(d2_list):
    d = {}
    for a, b in d2_list:
        # skip the non-variation last name instance
        if b == 0:
            continue
        if d.get(a) is None:
            d[a] = 0
        d[a] = d[a] + b
    d = sorted([[a, b] for a, b in d.items()], key=lambda x: x[0], reverse=False)
    return d


lastname_variants_vs_iagblock = aggregate_by_key(df_matched[['num_iag_in_block', 'varied_last_name']].values)
lastname_variants_vs_citationblock = aggregate_by_key(df_matched[['num_citation_in_block', 'varied_last_name']].values)


def plot_pub_year(whole_pub_year_dist, pub_year_counter, check_item):
    whole_pub_year_dist = [n for n in whole_pub_year_dist if 1970 <= int(n[0]) <= 2018]
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [int(n[0]) for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter = [n for n in pub_year_counter if 1970 <= n[0] <= 2018]
    pub_cnt = sum([n[1] for n in pub_year_counter])
    pub_year = [n[0] for n in pub_year_counter]
    pub_count = [n[1] * 1.0 / pub_cnt for n in pub_year_counter]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year, pub_count, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND', linewidth=linewidth)
    # plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('year', loc='right', fontsize=18)
    plot.ylabel('publication proportion (%)', loc='center', fontsize=18)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_author_position(whole_author_position_dist, author_position_counter, check_item):
    whole_pub_year_dist = [n for n in whole_author_position_dist if int(n[0]) <= 15]
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [int(n[0]) for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter = [n for n in author_position_counter if n[0] <= 15]
    pub_cnt = sum([n[1] for n in pub_year_counter])
    pub_year = [n[0] for n in pub_year_counter]
    pub_count = [n[1] * 1.0 / pub_cnt for n in pub_year_counter]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year, pub_count, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND', linewidth=linewidth)
    plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('position', loc='right', fontsize=18)
    plot.ylabel('author position proportion (%)', loc='center', fontsize=18)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_author_gender(whole_genni_gender_dist, author_genni_gender_counter, check_item):
    x_label_map = {'-': 'Unsure', 'F': 'Female', 'M': 'Male', '': ''}
    whole_pub_year_dist = sorted(whole_genni_gender_dist, key=lambda x: x[0], reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [x_label_map[n[0]] for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter = sorted([n for n in author_genni_gender_counter if x_label_map[n[0]] in set(whole_pub_year)],
                              key=lambda x: x[0], reverse=False)
    pub_cnt = sum([n[1] for n in pub_year_counter])
    pub_year = [x_label_map[n[0]] for n in pub_year_counter]
    pub_count = [n[1] * 1.0 / pub_cnt for n in pub_year_counter]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year, pub_count, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND', linewidth=linewidth)
    plot.title(check_item, fontsize=18)
    plot.xlabel('gender', loc='right', fontsize=18)
    plot.ylabel('author gender proportion (%)', loc='center', fontsize=18)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_ethnic_seer(whole_ethnic_seer_dist, author_ethnic_seer_counter, check_item):
    whole_pub_year_dist = sorted(whole_ethnic_seer_dist, key=lambda x: x[1], reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [n[0] for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    keys = [n[0] for n in author_ethnic_seer_counter]
    pub_year_counter = [author_ethnic_seer_counter[keys.index(n)] if n in keys else (n, 0) for n in whole_pub_year]
    pub_cnt = sum([n[1] for n in pub_year_counter])
    pub_year = [n[0] for n in pub_year_counter]
    pub_count = [n[1] * 1.0 / pub_cnt for n in pub_year_counter]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year, pub_count, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND', linewidth=linewidth)
    # plot.xscale('log')
    plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('ethnicity', loc='right', fontsize=18)
    plot.ylabel('ethnicity proportion (%)', loc='center', fontsize=18)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_ethnea(whole_ethnic_seer_dist, author_ethnic_seer_counter, check_item):
    whole_pub_year_dist = sorted(whole_ethnic_seer_dist, key=lambda x: x[1], reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [n[0] for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    keys = [n[0] for n in author_ethnic_seer_counter]
    pub_year_counter = [author_ethnic_seer_counter[keys.index(n)] if n in keys else (n, 0) for n in whole_pub_year]
    pub_cnt = sum([n[1] for n in pub_year_counter])
    pub_year = [n[0] for n in pub_year_counter]
    pub_count = [n[1] * 1.0 / pub_cnt for n in pub_year_counter]

    # plot.figure()
    idx = 0
    plot.loglog(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
                # marker=line_markers[idx], markersize=8, markevery=0.2,
                color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.loglog(pub_year, pub_count, linestyle=linestyles[idx],
                # marker=line_markers[idx], markersize=8, markevery=0.2,
                color=colors[idx], label='LAGOS-AND', linewidth=linewidth)
    plot.title(check_item, fontsize=18)
    plot.xlabel('ethnicity', loc='right', fontsize=18)
    plot.ylabel('ethnicity proportion (%)', loc='center', fontsize=18)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_lastname_popularity(whole_lastname_popularity_dist, author_lastname_counter, check_item):
    ratio = len(author_lastname_counter) * 1.0 / len(whole_lastname_popularity_dist)
    used_for_plot_ratio = 1
    whole_pub_year_dist = sorted([n for n in whole_lastname_popularity_dist],
                                 # if random() <= used_for_plot_ratio * ratio
                                 key=lambda x: x[0], reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [int(n[0]) for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter = sorted([n for n in author_lastname_counter],  # if random() <= used_for_plot_ratio
                              key=lambda x: x[0], reverse=False)
    pub_cnt = sum([n[1] for n in pub_year_counter])
    pub_year = [int(n[0]) for n in pub_year_counter]
    pub_count = [n[1] * 1.0 / pub_cnt for n in pub_year_counter]

    # plot.figure()
    idx = 0
    plot.scatter(whole_pub_year,  # [n * 100.0 / len(whole_pub_dist) for n in range(len(whole_pub_dist))],
                 whole_pub_dist,
                 marker='.',
                 color=colors[idx], label='MAG', s=4)
    idx = 1
    plot.scatter(pub_year,  # [n * 100.0 / len(pub_year) for n in range(len(pub_year))],
                 pub_count,
                 marker='o',
                 color=colors[idx], label='LAGOS-AND', s=4)
    plot.xscale('log')
    plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('LN', loc='right', fontsize=18)
    plot.ylabel('LN proportion (%)', loc='center', fontsize=18)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_namespace_popularity(whole_lastname_popularity_dist, author_lastname_counter, check_item):
    ratio = len(author_lastname_counter) * 1.0 / len(whole_lastname_popularity_dist)
    used_for_plot_ratio = 1
    whole_pub_year_dist = sorted([n for n in whole_lastname_popularity_dist],
                                 # if random() <= used_for_plot_ratio * ratio
                                 key=lambda x: x[0], reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [int(n[0]) for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    pub_year_counter = sorted([n for n in author_lastname_counter],  # if random() <= used_for_plot_ratio
                              key=lambda x: x[0], reverse=False)
    pub_cnt = sum([n[1] for n in pub_year_counter])
    pub_year = [int(n[0]) for n in pub_year_counter]
    pub_count = [n[1] * 1.0 / pub_cnt for n in pub_year_counter]

    # plot.figure()
    idx = 0
    plot.scatter(whole_pub_year,  # [n * 100.0 / len(whole_pub_dist) for n in range(len(whole_pub_dist))],
                 whole_pub_dist,
                 marker='.',
                 color=colors[idx], label='MAG', s=4)
    idx = 1
    plot.scatter(pub_year,  # [n * 100.0 / len(pub_year) for n in range(len(pub_year))],
                 pub_count,
                 marker='o',
                 color=colors[idx], label='LAGOS-AND', s=4)
    plot.xscale('log')
    plot.yscale('log')
    plot.title(check_item, fontsize=18)
    plot.xlabel('LNFI', loc='right', fontsize=18)
    plot.ylabel('LNFI proportion (%)', loc='center', fontsize=18)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_fos(whole_fos_dist, fos_counter, check_item):
    whole_pub_year_dist = sorted(whole_fos_dist, key=lambda x: x[1], reverse=False)
    all_pub_cnt = sum([n[1] for n in whole_pub_year_dist])
    whole_pub_year = [n[0] for n in whole_pub_year_dist]
    whole_pub_dist = [n[1] * 1.0 / all_pub_cnt for n in whole_pub_year_dist]

    keys = [n[0] for n in fos_counter]
    print(keys)
    pub_year_counter = [fos_counter[keys.index(n)] if n in keys else (n, 0) for n in whole_pub_year]
    pub_cnt = sum([n[1] for n in pub_year_counter])
    pub_year = [n[0] for n in pub_year_counter]
    pub_count = [n[1] * 1.0 / pub_cnt for n in pub_year_counter]

    # plot.figure()
    idx = 0
    plot.plot(whole_pub_year, whole_pub_dist, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='MAG', linewidth=linewidth)
    idx = 1
    plot.plot(pub_year, pub_count, linestyle=linestyles[idx],
              # marker=line_markers[idx], markersize=8, markevery=0.2,
              color=colors[idx], label='LAGOS-AND', linewidth=linewidth)
    # plot.yscale('log')
    plot.xticks(fontsize=10, rotation=45, ha='right')
    # plot.autofmt_xdate(bottom=0.2, rotation=30, ha='center')
    plot.title(check_item, fontsize=18)
    # plot.xlabel('domain', loc='right')
    plot.ylabel('domain proportion (%)', loc='center', fontsize=18)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_group_size(whole_fos_dist, check_item):
    whole_fos_dist = sorted(whole_fos_dist, key=lambda x: x[0], reverse=False)[1:]
    print(check_item, whole_fos_dist[0], whole_fos_dist[-1])
    idx = 1
    plot.scatter([n[0] for n in whole_fos_dist], [n[1] for n in whole_fos_dist],
                 color=colors[idx], label='LAGOS-AND', s=4)
    plot.xscale('log')
    plot.yscale('log')
    plot.title(check_item, fontsize=22)
    plot.xlabel('the number of groups', loc='right', fontsize=22)
    plot.ylabel('size of citation group', loc='center', fontsize=22)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_block_size(whole_fos_dist, check_item):
    print(check_item, whole_fos_dist[0], whole_fos_dist[-1])
    whole_fos_dist = sorted(whole_fos_dist, key=lambda x: x[0], reverse=False)
    idx = 1
    plot.scatter([n[0] for n in whole_fos_dist], [n[1] for n in whole_fos_dist],
                 color=colors[idx], label='LAGOS-AND', s=4)
    plot.xscale('log')
    plot.yscale('log')
    plot.title(check_item, fontsize=22)
    plot.xlabel('the number of blocks', loc='right', fontsize=22)
    plot.ylabel('block size', loc='center', fontsize=22)  # 'top'
    plot.legend(loc='best')  # 'lower right'


def plot_lastname_variants_in_block(whole_fos_dist, check_item, xlabel, ylabel):
    print(check_item, whole_fos_dist[0], whole_fos_dist[-1])
    whole_fos_dist = sorted(whole_fos_dist, key=lambda x: x[0], reverse=False)
    idx = 1
    plot.scatter([n[0] for n in whole_fos_dist], [n[1] for n in whole_fos_dist],
                 color=colors[idx], label='LAGOS-AND', s=4)
    plot.xscale('log')
    plot.yscale('log')
    plot.title(check_item, fontsize=22)
    plot.xlabel(xlabel, loc='right', fontsize=22)
    plot.ylabel(ylabel, loc='center', fontsize=22)  # 'top'
    plot.legend(loc='best')  # 'lower right'


plot.figure(42, figsize=(12, 18), dpi=300)
plot.subplot(421)
# plot.grid(True)
plot_pub_year(whole_pub_year_dist, pub_year_counter, check_item='(a) publication year distribution')
plot.subplot(422)
plot_author_position(whole_author_position_dist, author_position_counter, check_item='(b) author position distribution')
plot.subplot(423)

plot_author_gender(whole_genni_gender_dist, author_genni_gender_counter, check_item='(c) gender distribution')
plot.subplot(424)
# plot_author_gender(whole_mac_gender_dist, author_sex_mac_counter, check_item='mac_gender')
# plot_author_gender(whole_ssn_gender_dist, author_ssn_gender_counter, check_item='ssn_gender')

plot_ethnic_seer(whole_ethnic_seer_dist, author_ethnic_seer_counter, check_item='(d) ethnicity distribution')
plot.subplot(425)

# plot_ethnea(whole_ethnea_dist, author_ethnea_counter, check_item='(d) ethnicity distribution')
# plot.subplot(425)

plot_lastname_popularity(whole_lastname_popularity_dist, author_lastname_counter,
                         check_item='(e) last name (LN) distribution')
plot.subplot(426)
plot_namespace_popularity(whole_lastname_first_initial_popularity_dist, lastname_first_initial_counter,
                          check_item='(f) last name first initial (LNFI) distribution')
plot.subplot(427)
plot_fos(whole_fos_dist, fos_counter, check_item='(g) domain distribution')

plot.tight_layout()
plot.savefig('cached/gold-standard-check.png')
plot.show()

plot.figure(22, figsize=(16, 12), dpi=300)
plot.subplot(221)
plot_group_size(author_group_size_counter, check_item='(a) citation group size distribution')
plot.subplot(222)
plot_block_size(block_size_counter, check_item='(b) block size distribution')
plot.subplot(223)
plot_lastname_variants_in_block(lastname_variants_vs_citationblock, check_item='(c) last name variants distribution',
                                xlabel='block size',
                                ylabel='the number of last name variants')
# plot.subplot(224)
# plot_lastname_variants_in_block(lastname_variants_vs_iagblock, check_item='(d) last name variants distribution',
#                                 xlabel='block size',
#                                 ylabel='the number of last name variants')
plot.tight_layout()

plot.savefig('cached/dataset-size-distribution.png')

plot.show()
