import os

from scrapy import cmdline

# spider_path = os.path.abspath(__file__)
# 上次运行到网络工程师 page9
clash_path = os.path.abspath(r'..\clash-agent')
clash = os.path.join(clash_path, 'clash.meta.exe')
# 运行clash
os.system(rf'start cmd /K {clash} -d {clash_path}')

cmdline.execute('scrapy crawl main'.split())