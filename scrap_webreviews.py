# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:28:44 2017

@author: Gowtham Mallikarjuna
"""

import os
from bs4 import BeautifulSoup
import urllib.request

#os.chdir('C:\\Users\\gmallik\\Downloads\\delete\\python\\scraping')
url = 'https://www.amazon.com/Canon-Digital-Camera-Body-Black/product-reviews/B01BUYK04A/ref=cm_cr_arp_d_viewpnt_rgt?ie=UTF8&reviewerType=avp_only_reviews&filterByStar=critical&pageNumber=1'
r = urllib.request.urlopen(url).read()
soup = BeautifulSoup(r,'lxml')

text = soup.find_all("div",class_= "a-row review-data")

review_text=''
for i in text:
    #print(i.text)
    review_text+=' '+i.text


f = open('canon80d_cretical.txt','w+')
f.write(review_text)
f.close()
#print(review_text)
