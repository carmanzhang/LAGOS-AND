import csv
import logging as log
import os
import warnings

import joblib
from pandahouse import read_clickhouse

from client.clickhouse_client import CH, CHClient

log.getLogger().setLevel(log.INFO)

warnings.filterwarnings('ignore')

db_host = '127.0.0.1'
db_tcp_port = '9001'
db_http_port = '8124'
db_name = 'pr'
db_passwd = 'root'

conn = CH(host=db_host, http_port=db_http_port, passwd=db_passwd).get_conn()
tcp_client = CHClient().get_client()


class DBReader:
    @classmethod
    def cached_read(self, cached_file_path: str, sql: str, cached=True):
        if os.path.exists(cached_file_path):
            # df = pd.read_csv(cached_file_path, sep=',')
            df = joblib.load(cached_file_path)
            log.info('loaded raw dataset from local cache')
        else:
            df = read_clickhouse(sql, index=False, connection=conn, encoding='iso8859-1', quoting=csv.QUOTE_NONE)
            log.info('loaded raw dataset from database')
            if cached:
                joblib.dump(df, cached_file_path, compress=True)
                log.info('cached raw dataset into local directory')
                # df.to_csv(cached_file_path, index=False, header=True)
        return df

    @classmethod
    def tcp_model_cached_read(self, cached_file_path: str, sql: str, cached=True):
        if os.path.exists(cached_file_path):
            # df = pd.read_csv(cached_file_path, sep=',')
            df = joblib.load(cached_file_path)
            log.info('loaded raw dataset from local cache')
        else:
            df = tcp_client.query_dataframe(sql)
            log.info('loaded raw dataset from database')
            if cached:
                joblib.dump(df, cached_file_path, compress=True)
                log.info('cached raw dataset into local directory')
                # df.to_csv(cached_file_path, index=False, header=True)
        return df

    @classmethod
    def insert_dataframe(self, df, sql: str):
        tcp_client.insert_dataframe(query=sql, dataframe=df)
        log.info('loaded raw dataset from database')

