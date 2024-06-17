
import pandas as pd
import pymysql
import pymysql.cursors


"""
读数据->清洗数据->更新数据
"""
class Database:
    def __init__(self, host, port, user, password, database, charset):
        self.conn = pymysql.connect(host=host, port=port,
                            user=user, password=password,
                            database=database,
                            charset=charset)
        self.cursor = self.conn.cursor()

    def read_data(self):
        self.cursor.execute("SELECT * FROM `zp_boss`")



if __name__ == "__main__":
    # host = '127.0.0.1'
    # port = 3306
    # user = 'zp_boss'
    # password = 'zp_boss123456'
    # databse = 'zp_boss'
    # charset = 'utf8mb4'
    # DB = Database(host=host, port=port,
    #                 user=user, password=password,
    #                 database=databse, charset=charset)
    # DB.read_data()
    df = pd.read_csv('666.csv')
    company_names = list(df.company_name)
    company_area = list(df.company_area)
    job_names = list(df.job_name)
    details = list(df.detail)
    pass


    pass