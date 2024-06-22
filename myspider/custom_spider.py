import os
import subprocess

from lxml import etree
from playwright.sync_api import sync_playwright


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

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://localhost:{chrome_port}")
        context = browser.contexts[0]
        page = context.new_page()
        page.goto('https://www.zhipin.com')
        res = page.content()

        """
        爬取职位类型共25个，下属的职业的超链接以及名称，用于制作起始url
        """
        tree = etree.HTML(res)
        # 25个职业类型文本
        job_type_list = tree.xpath("//div[@class='home-job-menu']/div[1]/dl/dd/b/text()")
        # 因为最后一个职业类型是其他，直接舍去，故为24个职业类型
        for job in len(job_type_list):
            tree.xpath(f"//div[@class='home-job-menu']/div[1]/dl[{job+1}]")
        pass