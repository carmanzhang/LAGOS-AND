import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold

from myio.data_reader import DBReader

sql = """select * from and_ds.randomly_selected_block_content_for_visualization;"""
df = DBReader.tcp_model_cached_read('', sql=sql, cached=False)
ids = df['id'].to_numpy()
X = np.array(list(df['embedding'].apply(lambda x: list(x)).values), dtype=np.float)

temp = sorted([len(x) for x in X])
assert temp[0] == temp[-1]
y = df['target'].to_numpy()
# y_label_index = sorted(list(set(y)))
# y_label_index = {j: i for i, j in enumerate(y_label_index)}
print(X.shape)
colors = {'k': '#0334FB', 'p': '#FB3003', 'rk': '#D2D3F6', 'rp': '#F1DBE2'}
# markers = {'k': '+', 'p': '2', 'rk': '+', 'rp': '2'}
markers = {'k': 'o', 'p': 'X', 'rk': 'o', 'rp': 'X'}

'''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, metric='precomputed')
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

scatter_category_x = {'k': [], 'p': [], 'rk': [], 'rp': []}
scatter_category_y = {'k': [], 'p': [], 'rk': [], 'rp': []}
for i in range(X_norm.shape[0]):
    # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y_label_index[y[i]]),
    #          fontdict={'weight': 'bold', 'size': 9})
    # plt.scatter(X_norm[i, 0], X_norm[i, 1], s=80, c=colors[y[i]], marker=markers[y[i]])
    scatter_category_x[y[i]].append(X_norm[i, 0])
    scatter_category_y[y[i]].append(X_norm[i, 1])

plt.figure(figsize=(8, 8))
plt.xlabel('1th-D')
plt.ylabel('2nd-D')

subplot_0 = plt.scatter(scatter_category_x['k'], scatter_category_y['k'], s=60, c=colors['k'], marker=markers['k'])
subplot_1 = plt.scatter(scatter_category_x['rk'], scatter_category_y['rk'], s=60, c=colors['rk'], marker=markers['rk'])
subplot_2 = plt.scatter(scatter_category_x['p'], scatter_category_y['p'], s=60, c=colors['p'], marker=markers['p'])
subplot_3 = plt.scatter(scatter_category_x['rp'], scatter_category_y['rp'], s=60, c=colors['rp'], marker=markers['rp'])
plt.legend((subplot_0, subplot_1, subplot_2, subplot_3),
           (u'author mesh keyword', u'randomly selected mesh keyword', u'author cited paper',
            u'randomly selected paper'), loc='best')

plt.savefig('mtl-runnable-example.png', dpi=500)
plt.gcf().savefig('mtl-runnable-example.eps', format='eps', dpi=500)
plt.show()
