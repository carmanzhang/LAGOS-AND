from clickhouse_driver import Client


class CHClient:
    def __init__(self, host='localhost', user='default', pwd='root', port=9001, db='and_ds'):
        self.client = Client(host=host, user=user, database=db, password=pwd, port=port)

    def get_client(self):
        return self.client


class CH:
    def __init__(self, host='localhost', http_port='8124', db='and', user='default', passwd='root'):
        self.host = host
        self.http_port = http_port
        self.db = db
        self.user = user
        self.passwd = passwd

    def get_conn(self):
        self.conn = {'host': 'http://' + self.host + ':' + self.http_port,
                     'database': self.db,
                     'user': self.user,
                     'password': self.passwd}
        return self.conn
