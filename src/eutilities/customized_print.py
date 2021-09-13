import pandas as pd

pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置列名对齐
pd.set_option('display.unicode.east_asian_width', True)  # 设置列名对齐
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('expand_frame_repr', False)  # 设置不换行


def pprint(kv: list, decimal=2, pctg=False, sep=None):
    k = [item[0] for item in kv]
    if pctg:
        v = [round(item[1] * 100.0, decimal) for item in kv]
    else:
        v = [round(item[1], decimal) for item in kv]
    if not sep:
        df = pd.DataFrame(data=[v], columns=k)
        print(df.head())
    else:
        print(sep.join([str(s) for s in v]))
