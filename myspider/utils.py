import json
import random
import time

from lxml import etree
from playwright.sync_api import sync_playwright

def use_chrome(start=False):
    if start:
        p = sync_playwright().start()
    else:
        p.stop()
    browser = p.chromium.launch(headless=False, slow_mo=50, proxy={"server": "http://192.168.123.254:7890"})
    context = browser.new_context()

    # 隐藏特征（反爬虫）
    context.add_init_script(path="stealth.min.js")

    page = context.new_page()
    # 加载cookies
    with open('cookies.json', 'r', encoding='utf-8') as file:
        cookies = json.loads(file.read())
        context.add_cookies(cookies=cookies)
    page.goto('https://zhipin.com')
    return page

def get_cookies():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, proxy={"server": "http://192.168.123.254:7890"})
        context =  browser.new_context()

        # 隐藏特征（反爬虫）
        context.add_init_script(path="myspider/stealth.min.js")

        page = context.new_page()
        page.goto('https://zhipin.com')
        page.wait_for_selector("//div[@class='job-menu-wrapper']")
        # 当登录完成后存在cookies数据再写入cookies.json文件
        input("是否登录完成？\n：")
        page.goto('https://zhipin.com')
        # 获取cookies数据写入文件
        with open('myspider/cookies.json', 'w', encoding='utf-8') as file:
            json.dump(context.cookies(), file)
            print("cookies写入完成!")
        # 不要关闭浏览器，关闭浏览器cookies将会失效
        input("是否关闭浏览器？\n：")

def get_job_type():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, proxy={"server": "http://192.168.123.254:7890"})
        context =  browser.new_context()

        # 隐藏特征（反爬虫）
        context.add_init_script(path="myspider/stealth.min.js")

        page = context.new_page()
        page.goto('https://zhipin.com')
        lxml_class = etree.HTML(page.content()).xpath("//div[@class='job-menu-wrapper']//div[@class='menu-sub']//div[@class='text']/a/text()")
        with open("myspider/job_type.txt", 'w', encoding='utf-8') as file:
            for i in lxml_class:
                file.write(str(i))
                file.write('\n')

if __name__ == "__main__":
    # get_job_type()
    get_cookies()