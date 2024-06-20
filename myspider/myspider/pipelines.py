# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from ast import main
from itemadapter import ItemAdapter
from scrapy.crawler import Crawler
import pymysql


class MyspiderPipeline:

    @classmethod
    def from_crawler(cls, crawler: Crawler):
        host = crawler.settings['DB_HOST']
        port = crawler.settings['DB_PORT']
        username = crawler.settings['DB_USER']
        password = crawler.settings['DB_PASS']
        database = crawler.settings['DB_NAME']
        return cls(host, port, username, password, database)

    def __init__(self, host, port ,username, password, database):
        self.conn = pymysql.connect(host=host, port=port,
                                    user=username, password=password,
                                    database=database, charset='utf8mb4')
        self.cursor = self.conn.cursor()
        self.data =[]

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        if len(self.data) > 0:
            self._write_to_db()
            self.data=[]
        self.conn.close()

    def process_item(self, item, spider):
        company_name = item.get('company_name', '')
        job_name = item.get('job_name', '')
        company_area = item.get('job_area', '')
        detail = item.get('detail', '')
        self.data.append((company_name, company_area, job_name, detail))
        print("当前已经积累item数量为：", len(self.data), "个")
        if len(self.data) >= 10:
            self._write_to_db()
            self.data=[]
        return item

    def _write_to_db(self):
        """
        数据以元组类型为一组进行execute， 假如是executemany的话，list里边一组数据也是一个元组
        """
        sql = 'insert into zp_boss (company_name, company_area, job_name, detail) values (%s, %s, %s, %s)'
        self.cursor.executemany(sql, self.data)
        self.conn.commit()