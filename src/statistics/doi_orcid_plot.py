import seaborn as sb

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sb.set_theme(style="ticks", rc=custom_params)

from matplotlib import pyplot as plot

plot.rcParams['font.family'] = 'serif'
plot.rcParams['font.serif'] = ['Times New Roman'] + plot.rcParams['font.serif']

from myio.data_reader import DBReader

colors = ['green', 'red', 'gold', 'black', 'cyan', 'blue', 'magenta', 'purple', 'gray', 'fuchsia', 'orange', 'yellow']
linestyles = ['--', '-.', '--', '--']
line_markers = ['<', '>', '^', 'v']
linewidth = 5

df_orcid = DBReader.tcp_model_cached_read("cached/num_orcid_each_year.pkl",
                                          """select toString(JSONExtractInt(
                                                    JSONExtractRaw(submission_date, 'value') as submission_date_tmp,
                                                    'year')) as year,
                                                   count()   as cnt
                                            from orcid.orcid
                                            group by year
                                            order by year
                                            ;""",
                                          cached=False)

df_doi = DBReader.tcp_model_cached_read("cached/num_doi_each_year.pkl",
                                        """select year, count() as cnt
                                          from (
                                                select issued == 'null' ? '': splitByChar('-', issued)[1] as year
                                                from doi.crossref_doiboost
                                                where length(year) > 0)
                                          group by year
                                          having year >= '1980'
                                             and year <= '2019'
                                          order by year
                                          ;""",
                                        cached=False)

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
plot.xlabel('year', loc='right')

plot.ylabel('number of new records', loc='center')  # 'top'
plot.legend(loc='best')  # 'lower right'

plot.tight_layout()
plot.savefig('cached/doi_orcid_each_year.png', dpi=600)

plot.show()
