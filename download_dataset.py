from icrawler.builtin import BingImageCrawler

plants = {
    "monstera plant": "dataset/monstera",
    "aloe vera plant": "dataset/aloe_vera",
    "lily plant": "dataset/lily"
}

for keyword, folder in plants.items():
    crawler = BingImageCrawler(storage={"root_dir": folder})
    crawler.crawl(keyword=keyword, max_num=150)

print("Download finished")
