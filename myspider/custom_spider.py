import json
import os
import random
import subprocess
import time
from urllib.parse import urljoin

from fake_useragent import UserAgent
from itertools import zip_longest
from lxml import etree
import pandas as pd
from playwright.sync_api import sync_playwright


def get_job_type_to_json(res, flag):
    if flag:
        """
            爬取职位类型共25个，下属的职业的超链接以及名称，用于制作起始url
            """
        tree = etree.HTML(res)
            # 25个职业类型文本
        job_type_list = tree.xpath("//div[@class='home-job-menu']/div[1]/dl/dd/b/text()")
            # 初始化空字典
        my_dict = {}
        base_url = 'https://www.zhipin.com'
            # 因为最后一个职业类型是其他，直接舍去，故为24个职业类型
        for job in range(len(job_type_list)-1):
            key = str(tree.xpath(f"//div[@class='home-job-menu']/div[1]/dl[{job+1}]/dd/b/text()")[0])
            data = tree.xpath(f"//div[@class='home-job-menu']/div[1]/dl[{job+1}]/div/ul/li/div/a/text()")
            data_url = tree.xpath(f"//div[@class='home-job-menu']/div[1]/dl[{job+1}]/div/ul/li/div/a/@href")
            result = [a + '&&' + urljoin(base_url, b) for a, b in zip_longest(data, data_url, fillvalue=0)]
            my_dict[f'{key}']=result
        json_data = json.dumps(my_dict, ensure_ascii=False)
        with open('myspider/job_type.json', 'w', encoding='utf-8')as f:
            f.write(json_data)
        print("工作种类爬取成功")

if __name__ == "__main__":
    """
    打开浏览器，在浏览器目录下打开命令行 '.\chrome.exe --remote-debugging-port=9999'
    """
    chrome_port = 9999

    # # open the local Chrome
    # chrome_path = r'"C:\Program Files\Google\Chrome\Application\chrome.exe"'
    # debugging_port = f"--remote-debugging-port={chrome_port}"
    # command = f"{chrome_path} {debugging_port}"
    # subprocess.Popen(command, shell=True)
    ua = UserAgent()
    # proxies = [
    #     'http://127.0.0.1:10110',
    #     'http://127.0.0.1:10111',
    #     'http://127.0.0.1:10112',
    #     'http://127.0.0.1:10113',
    #     'http://127.0.0.1:10114'
    # ]
    proxies = [
        'http://127.0.0.1:10809',
        'http://127.0.0.1:10809'
    ]
    def random_proxy():
        return random.choices(proxies)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50, proxy={"server": str(random_proxy()[0])})
        with browser.new_context(proxy={'server': str(random_proxy())}, extra_http_headers={'User-Agent': str(ua.random())}) as context:
            # 隐藏特征（反爬虫）
            context.add_init_script(path="stealth.min.js")
            page = context.new_page()

            page.goto("https://www.zhipin.com")
            res = page.content()

        get_job_type_to_json(res, flag=False)

        with open('myspider/job_type.json', 'r', encoding='utf-8') as f:
            job_data = json.loads(f.read())
        job_keys = list(job_data.keys())
        for i in job_keys:
            print(f"正在爬取 {i} 的内容")
            for j in job_data[i]:
                j = j.split("&&")
                print(f"正在爬取 {j[0]} 这个岗位")
                page.wait_for_timeout(random.random()*10000)
                # 这里只爬取一页数据，不过也有30个公司了
                page.goto(j[1])
                job_res = page.content()
                tree = etree.HTML(job_res)
                job_name_list = tree.xpath("//div[@class='job-list']/ul/li//div[@class='job-title']/span[1]/a/text()")
                job_area_list = tree.xpath("//div[@class='job-list']/ul/li//div[@class='job-title']/span[2]/span/text()")
                company_list = tree.xpath("//div[@class='job-list']/ul/li//div[@class='info-company']//h3/a/text()")
                details_list = []
                for k in range(len(job_name_list)):
                    detail_list = tree.xpath(f"//div[@class='job-list']/ul/li[{k+1}]//div[@class='tags']/span/text()")
                    detail_list = ','.join(detail_list)
                    details_list.append(detail_list)
                result = [a+'__'+b+'__'+c+'__'+d for a, b, c, d in zip_longest(job_name_list, job_area_list, company_list, details_list, fillvalue=0)]
                with open('myspider/train.txt', 'a+', encoding='utf-8')as ff:
                    for l in result:
                        ff.write(l+'\n')