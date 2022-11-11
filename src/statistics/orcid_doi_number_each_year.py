import os

import seaborn as sb

from myconfig import cached_dir, latex_doc_base_dir

# sb.set_style("darkgrid")
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sb.set_theme(style="ticks", rc=custom_params)

from matplotlib import pyplot as plot

plot.rcParams['font.family'] = 'serif'
plot.rcParams['font.serif'] = ['Times New Roman'] + plot.rcParams['font.serif']

from mytookit.data_reader import DBReader

colors = ['green', 'red', 'gold', 'black', 'cyan', 'blue', 'magenta', 'purple', 'gray', 'fuchsia', 'orange', 'yellow']
linestyles = ['--', '-.', '--', '--']
line_markers = ['<', '>', '^', 'v']
linewidth = 5

df_orcid = DBReader.tcp_model_cached_read(os.path.join(cached_dir, "num_orcid_each_year.pkl"),
                                          "", cached=True)
print('df_orcid: ', df_orcid.values)

df_doi = DBReader.tcp_model_cached_read(os.path.join(cached_dir, "num_doi_each_year.pkl"), "", cached=True)

print('df_doi: ', df_doi.values)

plot.figure()
# plot.grid(which='major', axis='y')
idx = 0
plot.plot(df_doi.values[:, 0].astype('int'), df_doi.values[:, 1], linestyle=linestyles[idx],
          # marker=line_markers[idx], markersize=8, markevery=0.2,
          color=colors[idx], label='DOI', linewidth=linewidth)

idx = 1
plot.plot(df_orcid.values[:, 0].astype('int'), df_orcid.values[:, 1], linestyle=linestyles[idx],
          # marker=line_markers[idx], markersize=8, markevery=0.2,
          color=colors[idx], label='ORCID', linewidth=linewidth)

# plot.yscale('log')
# plot.title('num of instance each year')
# plot.xlabel('year', loc='right')

plot.ylabel('# Records', loc='center', fontsize=18)  # 'top'
plot.legend(loc='best')  # 'lower right'
plot.xticks(fontsize=18)
plot.yticks(fontsize=18)

plot.tight_layout()
plot.savefig(os.path.join(cached_dir, 'doi_orcid_each_year.png'), dpi=600)
plot.savefig(os.path.join(latex_doc_base_dir, 'figs/doi_orcid_each_year.png'), dpi=600)

plot.show()
