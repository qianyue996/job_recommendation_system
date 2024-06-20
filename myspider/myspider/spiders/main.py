import time
from urllib.parse import unquote, quote, urljoin

from lxml import etree
import scrapy
from sympy import hermite

from myspider.items import MyspiderItem


class DemoSpider(scrapy.Spider):
    name = "main"
    allowed_domains = ["zhipin.com"]
    base_url = 'https://zhipin.com'
    def __init__(self):
        with open('job_type.txt', 'r', encoding='utf-8') as file:
            self.job_names = file.readlines()[80:][:-1]

    def start_requests(self):
        for job in self.job_names:
            job = quote(job.strip(), 'utf-8')
            for page in range(0, 10):
                with open('running_log.log', 'a+', encoding='utf-8') as f:
                    f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} | 当前运行到的工作名称为：{unquote(job)}, 第{page+1}页\n')
                yield scrapy.Request(
                    url=f"https://zhipin.com/web/geek/job?query={job}&city=101030100&page={page+1}",
                    meta={'counter':page}
                )

    def parse(self, response, **kwargs):
        item = MyspiderItem()
        job_list = response.xpath("//ul[@class='job-list-box']/li").getall()
        for i in job_list:
            tree = etree.HTML(i)
            detail_url = urljoin(self.base_url, tree.xpath("//li/div[1]/a/@href")[0])
            item['job_name'] = tree.xpath("//span[@class='job-name']/text()")[0]
            item['company_name'] = tree.xpath("//li//h3[@class='company-name']/a/text()")[0]
            item['job_area'] = tree.xpath("//span[@class='job-area-wrapper']/span/text()")[0]
            yield scrapy.Request(url=detail_url, callback=self.parse_detail, cb_kwargs={'item':item}, priority=1)

    def parse_detail(self, response, **kwargs):
        item = kwargs['item']
        item['detail'] = ''.join(response.xpath("//div[@class='job-sec-text']/text()").getall())
        yield item
