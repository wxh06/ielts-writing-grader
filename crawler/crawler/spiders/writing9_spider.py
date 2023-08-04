import json

import scrapy
from scrapy.linkextractors import LinkExtractor


class Writing9Spider(scrapy.Spider):
    name = "writing9"
    start_urls = ("https://writing9.com/ielts-writing-samples",)

    def parse(self, response):
        for link in LinkExtractor(
            allow=r"https:\/\/writing9\.com\/text\/"
        ).extract_links(response):
            yield scrapy.Request(url=link.url, callback=self.parse_sample)

    def parse_sample(self, response):
        props = json.loads(response.css("#__NEXT_DATA__::text").get())["props"][
            "pageProps"
        ]
        yield {
            "question": props["text"]["question"].replace("\r\n", "\n"),
            "essay": props["text"]["text"].replace("\r\n", "\n"),
            "band": props["results"]["bands"]["band"],
        }
