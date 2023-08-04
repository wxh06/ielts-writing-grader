import json

import scrapy
from scrapy.linkextractors import LinkExtractor


class Writing9Spider(scrapy.Spider):
    name = "writing9"
    start_urls = ("https://writing9.com/ielts-writing-samples",)
    slugs = []

    def parse(self, response):
        for link in LinkExtractor(
            allow=r"^https:\/\/writing9\.com\/text\/"
        ).extract_links(response):
            yield scrapy.Request(url=link.url, callback=self.parse)

        try:
            props = json.loads(response.css("#__NEXT_DATA__::text").get())["props"][
                "pageProps"
            ]
            if props["id"] not in self.slugs:
                self.slugs.append(props["id"])
                yield props
        except KeyError:
            pass
