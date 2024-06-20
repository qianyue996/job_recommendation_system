# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

import json
import random
import time

from scrapy import signals
from scrapy.crawler import Crawler
from scrapy.http.response.html import HtmlResponse
from playwright.sync_api import sync_playwright, expect

# useful for handling different item types with a single interface
from itemadapter import is_item, ItemAdapter

class MyspiderSpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)

class MyspiderDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        uas = crawler.settings.get('USER_AGENTS', None)
        proxies = crawler.settings.get('CLASH_AGENT', None)
        s = cls(uas, proxies)
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def __init__(self, uas, proxies):
        self.uas = uas
        self.proxies = proxies
        self.p = sync_playwright().start()
        self.browser = self.p.chromium.launch(headless=False, slow_mo=50, proxy={"server": "http://127.0.0.1:10111"})
        self.context = self.browser.new_context()

        # 隐藏特征（反爬虫）
        self.context.add_init_script(path="stealth.min.js")

        self.page = self.context.new_page()

        # 加载cookies
        # with open('cookies.json', 'r', encoding='utf-8') as file:
        #     cookies = json.loads(file.read())
        #     self.context.add_cookies(cookies=cookies)

        self.page.goto('https://zhipin.com')
        self.flag = False

    def init_state(self):
        ua = self.random_user_agent()
        proxy = self.random_clash_proxy()
        self.context = self.browser.new_context(proxy={'server':f'{proxy}'}, extra_http_headers={'User-Agent':f'{ua}'})
        # 隐藏特征（反爬虫）
        self.context.add_init_script(path="stealth.min.js")
        self.page = self.context.new_page()
        self.page.goto('https://zhipin.com')
        expect(self.page.locator("//div[@class='job-menu-wrapper']")).to_be_visible(timeout=10000)

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        page = request.meta.get('counter', None)
        if page == 0 and self.flag:
            self.page.close()
            self.init_state()

        self.page.wait_for_timeout(random.randint(1000, 4000))
        if 'job_detail' in request.url:
            self.page.goto(request.url)
            expect(self.page.locator("//div[@class='job-sec-text']")).to_be_visible(timeout=10000)
            self.page.on
            self.is_safe_verify(self.page, request, spider)
            return HtmlResponse(url=request.url,body=self.page.content(),request=request,encoding='utf-8')
        else:
            self.page.goto(request.url)
            expect(self.page.locator("//ul[@class='job-list-box']")).to_be_visible(timeout=10000)
            self.page.on
            self.is_safe_verify(self.page, request, spider)
            if self.flag == False:
                self.flag = True
            return HtmlResponse(url=request.url,body=self.page.content(),request=request,encoding='utf-8')


    def is_safe_verify(self, page, request, spider):
        if 'safe/verify' in page.url:
            input("是否解除验证？：")
            self.process_request(self, request, spider)

    def random_user_agent(self):
        return random.choice(self.uas)

    def random_clash_proxy(self):
        return random.choice(self.proxies)

    def process_response(self, request, response, spider):
    # Called with the response returned from the downloader.

    # Must either;
    # - return a Response object
    # - return a Request object
    # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)

    def __del__(self):
        self.p.stop()
