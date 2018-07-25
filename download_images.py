import requests
from functools import partial
import time
import urllib.request
import argparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from multiprocessing import Pool
from lxml.html import fromstring
import os

def search(url):
    options = Options()
    options.add_argument('headless')
    options.add_argument('window-size=1920,1080')
    options.add_argument('disable-gpu')
    options.add_argument('log-level=3')

    # Create a browser
    browser = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver',0,options)
    wait = WebDriverWait(browser,10)
    
    # Open the link
    browser.get(url)
    time.sleep(1)

    element = browser.find_element_by_tag_name("body")

    # Scroll down
    for i in range(2):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)

    browser.find_element_by_id("smb").click()

    for i in range(50):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)

    time.sleep(1)

    # Get page source and close the browser
    source = browser.page_source
    browser.close()

    return source

def download_image(dest, link):
    # Use a random user agent header
    headers = {"User-Agent": ua.random}

    # Get the image link
    try:
        r = requests.get("https://www.google.com" + link, headers=headers)
    except:
        print("Cannot get link.")
    title = str(fromstring(r.content).findtext(".//title"))
    link = title.split(" ")[-1]

    # Download the image
    print("Downloading from " + link)
    try:
        filename = urllib.request.urlretrieve(link, dest + "/" + link.split("/")[-1])
    except:
        pass

if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("keyword", help="the keyword to search")
    parser.add_argument("--worker", help="the number of workers used for downloading images",
            type=int, default=10)
    args = parser.parse_args()

    # get user input and search on google
    query = args.keyword
    url = "https://www.google.com/search?as_st=y&tbm=isch&as_q=" + query + \
          "&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=isz:lt,islt:svga,itp:photo,ift:jpg"
    source = search(url)

    # Parse the page source and download pics
    soup = BeautifulSoup(str(source), "html.parser")
    ua = UserAgent()

    # check directory and create if necessary
    dest = "images/" + args.keyword
    if not os.path.isdir(dest):
        os.makedirs(dest)

    # get the links
    links = soup.find_all("a", class_="rg_l")
    links = [link.get("href") for link in links]

    # open some processes to download
    with Pool(processes=args.worker) as pool:
        func = partial(download_image, dest)
        pool.map(func, links)
