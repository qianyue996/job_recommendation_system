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
        for key in job_type_list[:-1]:
            my_dict[f'{key}'] = {}
        base_url = 'https://www.zhipin.com'
            # 因为最后一个职业类型是其他，直接舍去，故为24个职业类型
        for job1 in range(len(job_type_list)-1):
            key1 = str(tree.xpath(f"//div[@class='home-job-menu']/div[1]/dl[{job1+1}]/dd/b/text()")[0])
            job2_type_list = tree.xpath(f"//div[@class='home-job-menu']/div[1]/dl[{job1+1}]/div/ul/li/h4/text()")
            for job2 in range(len(job2_type_list)):
                key2 = tree.xpath(f"//div[@class='home-job-menu']/div[1]/dl[{job1+1}]/div/ul/li/h4/text()")[job2]
                data = tree.xpath(f"//div[@class='home-job-menu']/div[1]/dl[{job1+1}]/div/ul/li[{job2+1}]/div/a/text()")
                data_url = tree.xpath(f"//div[@class='home-job-menu']/div[1]/dl[{job1+1}]/div/ul/li[{job2+1}]/div/a/@href")
                result = [a + '&&' + urljoin(base_url, b) for a, b in zip_longest(data, data_url, fillvalue=0)]
                my_dict[f'{key1}'][f'{key2}']=result
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
        # with browser.new_context(proxy={'server': str(random_proxy()[0])}, extra_http_headers={'User-Agent': str(ua.windows)}) as context:
        with browser.new_context() as context:
            # 隐藏特征（反爬虫）
            context.add_init_script(path="stealth.min.js")
            page = context.new_page()

            page.goto("https://www.zhipin.com/tianjin")
            res = page.content()

            get_job_type_to_json(res, flag=False)

            with open('myspider/job_type.json', 'r', encoding='utf-8') as f:
                job_data = json.loads(f.read())
            job_keys = list(job_data.keys())
            for i in job_keys:
                print(f"正在爬取 {i} 的内容")
                job_key2 = list(job_data[i].keys())
                for j in job_key2:
                    print(f"正在爬取 {i}-{j} 的内容")
                    job_key3 = job_data[i][j]
                    for k in job_key3:
                        k = k.split("&&")
                        print(f"正在爬取 {i}-{j}-{k[0]} 这个岗位")
                        page.wait_for_timeout(random.random()*4000)
                        # 这里只爬取一页数据，不过也有30个公司了

                        # https://www.zhipin.com/web/geek/job?query=&city=101030100&position=100101
                        reurl = k[1].split('/')[-2].split('-')
                        reurl_key1 = reurl[0].replace('c', 'city=')
                        reurl_key2 = reurl[1].replace('p', 'position=')
                        reurl = f'https://www.zhipin.com/web/geek/job?query=&{reurl_key1}&{reurl_key2}'
                        page.goto(reurl)
                        page.wait_for_selector(selector="//ul[@class='job-list-box']", timeout=10000)
                        job_res = page.content()
                        tree = etree.HTML(job_res)
                        job_name_list = tree.xpath("//ul[@class='job-list-box']/li//span[@class='job-name']/text()")
                        job_area_list = tree.xpath("//ul[@class='job-list-box']/li//span[@class='job-area']/text()")
                        company_list = tree.xpath("//ul[@class='job-list-box']/li//h3/a/text()")
                        details_list = []
                        for l in range(len(job_name_list)):
                            detail_list = tree.xpath(f"//ul[@class='job-list-box']/li[{l+1}]/div[2]//ul[@class='tag-list']/li/text()")
                            detail_list = ','.join(detail_list)
                            details_list.append(detail_list)
                        xx = [i] * len(job_name_list)
                        yy = [j] * len(job_name_list)
                        zz = [k[0]] * len(job_name_list)
                        result = [x+'__'+y+'__'+z+'__'+a+'__'+b+'__'+c+'__'+d for x, y, z, a, b, c, d in zip_longest(xx, yy, zz, job_name_list, job_area_list, company_list, details_list, fillvalue=0)]
                        with open('myspider/train.txt', 'a+', encoding='utf-8')as ff:
                            for l in result:
                                ff.write(l+'\n')