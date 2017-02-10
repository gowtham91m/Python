# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:28:44 2017

@author: Gowtham Mallikarjuna
"""

import os
from bs4 import BeautifulSoup
import urllib.request
import random
import time
import requests

url = 'web_url'
page = requests.get(url)
page=page.text
bs_page = BeautifulSoup(page,'lxml')
li=bs_page.findAll('li','page-button')
page=li[-1].text
#print(page)

text=[]
for txt in bs_page.find_all('span','a-size-base review-text'): text.append(txt.text)

for i in range(2,int(page)+1,1):
    x=random.randrange(7,40)
    time.sleep(x)
    url=url[:-1]+str(i)
    content = requests.get(url)
    content = content.text
    bs_content = BeautifulSoup(content,'lxml')
    for txt in bs_content.find_all('span','a-size-base review-text'):text.append(txt.text)

print(len(text))


'''
import urllib2
opener = urllib2.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
response = opener.open(url)
html_contents = response.read()
'''

l=len(text)
f = open('review.txt','a')

for i in range(len(text)):
    print(i)
    try:
        f.write(text[i])
        f.write('\n\n')
    except Exception as e:
        print(e)
        continue
f.close()

