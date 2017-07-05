import pandas as pd
import numpy as np

import dateutil
import datetime
import time
from time import sleep

from bs4 import BeautifulSoup
import requests
import random
import re
import html5lib
import html
import pickle as pk
from fake_useragent import UserAgent
import selenium

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
import os

def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, 'took', end - start, 'seconds'
        return result
    return f_timer


chromedriver = '/Users/feiwang/Documents/python_code/Petition Prediction/chromedriver'
os.environ['webdriver.chrome.driver'] = chromedriver
#chromedriver
driver = webdriver.Chrome(chromedriver)

# go straight to most-recent pages
petitions_url = 'https://www.change.org/petitions#most-recent/' # up to 5084

@timefunc
def get_petition_urls(base, start, stop):
    d = {}
    for i in range(start, stop+1):
        driver.get(base + str(i))
        petitions = driver.find_elements_by_xpath('//div[@class = "petition-list"]//ol//li[@class = "petition"]')
        for p in petitions:
            d[p.get_attribute('data-id')] = p.get_attribute('data-url')
    return d

def chunk_petition_urls(dic, base, first, last, chunk):
    for c in range(first, last+1, chunk):
        d = get_petition_urls(base, c, c+chunk)
        dic.update(d)
        filename = 'master' + str(c+chunk) + '.pkl'
        with open(filename, 'wb') as f:
            pk.dump(dic,f,-1)
    return dic

victories_url = 'https://www.change.org/victories#most-recent/' 

def get_victory_urls(base, start, stop):
    d = {}
    for i in range(start, stop+1):
        driver.get(base + str(i))
        sleep(random.randint(1,3))
        victories = driver.find_elements_by_xpath('//div[@class = "petition-list"]//ol//li[@class = "petition"]')
        for i, v in enumerate(victories):
            d[i] = v.get_attribute('data-url')
    return d

def chunk_victory_urls(dic, base, first, last, chunk):
    for c in range(first, last+1, chunk):
        d = get_victory_urls(base, c, c+chunk)
        dic.update(d)
        filename = 'victories' + str(c+chunk) + '.pkl'
        with open(filename, 'wb') as f:
            pk.dump(dic,f,-1)
    return dic


web_vic_dict=dict()
def get_petittion_info(url):    
    sleep(random.randint(1,2))
    one_dict=dict()
    driver.get(url)
    
    try:
        titles=driver.find_element_by_tag_name('h1').text
    except NoSuchElementException:
        return one_dict
    print titles
    
    try:
        total_supporter=driver.find_element_by_xpath \
        ('//div[@class = "col-xs-4 type-s js-mobile-supporter-count"]').text
    except NoSuchElementException:
        return one_dict   
    print total_supporter
    
    try:
        text=driver.find_element_by_xpath \
        ('//div[@class = "rte js-description-content"]').text
    except NoSuchElementException:
        one_dict[url]=[titles,total_supporter,[],[]]
        return one_dict
    
    sleep(random.randint(1,3))
    driver.execute_script("window.scrollTo(0,1000000000);")
    
    try:
        supporters_progression=driver.find_elements_by_xpath \
        ('//div[@class = "media mvn type-s type-highlight"]')
    except NoSuchElementException:
        one_dict[url]=[titles,text,total_supporter,[],[]]
        return one_dict
    sup_prog=[]
    for supporter in supporters_progression:
        sup_prog.append(supporter.text)

    try:
        starting_date=driver.find_element_by_xpath \
        ('//div[@class = "box box-basic bg-default"]').text
    except NoSuchElementException:
        starting_date = []
    print starting_date
    one_dict[url]=[titles,text,total_supporter,sup_prog,starting_date]
    
    return one_dict


    
def assemble_pettion_info(master_storage,url_dict,start_key):
    chunck_dict=dict()
    i=0
    print 'from'+ str(start_key)+ 'url'
    for k in xrange(start_key,len(url_dict)):
        one_dict=get_petittion_info(url_dict[k])
        chunck_dict.update(one_dict)
        master_storage.update(one_dict)
        i+=1
        print i
        
    pk.dump(chunck_dict, \
            open('vic_pettion_info_dict'+str(start_key)+'-'+str(start_key+i)+'_400.pk','wb'))
    pk.dump(master_storage, open('vic_pettion_info_master_dict.pk','wb'))    
    
    return chunck_dict
