# number of publication in each year
from matplotlib import pyplot as plt

from src.myio.data_reader import DBReader

num_pub_yearly_sql = 'select year, num_pubs from and.num_pub_yearly;'
num_cum_malformed_name_sql = 'select year, num_abbr_first_name from and.num_malformed_name_cumulatively;'
num_cum_author_with_aff_sql = 'select year, num_authors, num_authors_with_aff from and.num_author_with_aff_cumulatively;'

# print(df.shape)
# year, num_pubs = df['year'], df['num_pubs']
# pyplot.plot(year, num_pubs, '-')
# pyplot.show()

f, ax = plt.subplots(2, 2)

style_list = ["g+-", "r*-", "b.-", "yo-"]
df = DBReader.cached_read("./fsdgvfdsbfd", num_pub_yearly_sql, cached=False)
ax[0][0].plot(df['year'], df['num_pubs'], style_list[0])
df = DBReader.cached_read("./fsdgvfdsbfd", num_cum_malformed_name_sql, cached=False)
ax[0][1].plot(df['year'], df['num_abbr_first_name'], style_list[1])
df = DBReader.cached_read("./fsdgvfdsbfd", num_cum_author_with_aff_sql, cached=False)
ax[1][0].plot(df['year'], df['num_authors'], style_list[2])
ax[1][1].plot(df['year'], df['num_authors_with_aff'], style_list[3])

plt.savefig('motivation.png')
plt.show()
