import scrapy
import re
class ArticleSpider(scrapy.Spider):
    name='article'

    def start_requests(self):
        urls = [
            'https://feng.li/teaching/statcase/']
        return [scrapy.Request(url=url, callback=self.parse) for url in urls]

    def parse(self, response):
        url = response.url
        header = response.xpath('//*[@id="post-1101"]/header/h1').extract_first()
        print('\n'*2,'****'*8, '这是爬取的结果')
        print('URL is: {}'.format(url))
        print('Header is: {}'.format(header))
        print('-----------------------------------这是结束线', '\n'*2)

