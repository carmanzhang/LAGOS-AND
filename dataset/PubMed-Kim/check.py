from unidecode import unidecode

from myio.data_reader import DBReader

df = DBReader.tcp_model_cached_read("cached/YYYY",
                                    """select PMID,
                                    trimBoth(splitByString(',', MEDLINE_Name)[1]) as medline_lastname,
                                               splitByChar('_', AINI)[1] as block_lastname
                                        from and_ds.AUT_ORC
                                        where medline_lastname != block_lastname""",
                                    cached=False)
df['medline_lastname_parsed'] = df['medline_lastname'].apply(unidecode)
df['block_lastname_parsed'] = df['block_lastname'].apply(unidecode)

all = df.shape[0]
cnt = 0
for i, (pmid, medline_lastname, medline_lastname_parsed, block_lastname, block_lastname_parsed) in df[
    ['PMID', 'medline_lastname', 'medline_lastname_parsed', 'block_lastname', 'block_lastname_parsed']].iterrows():
    # medline_lastname_parsed = medline_lastname_parsed.lower().replace('-', '').replace(' ', '').replace('\'', '').replace('?', '')
    medline_lastname_parsed = ''.join([n for n in medline_lastname_parsed.lower() if n not in ('-',' ','\'', '?')])
    block_lastname_parsed = block_lastname_parsed.lower()
    if medline_lastname_parsed != block_lastname_parsed:
        print(pmid, medline_lastname_parsed, block_lastname_parsed)
        cnt += 1

print(cnt, all, cnt / all)
