
# coding: utf-8

# In[1]:


import os
#import pandas as pd
import operator
import re
import random

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.api import StringTokenizer
#from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
#from nltk.corpus import stopwords

#os.chdir('C:\\Users\\gmallik\\Downloads\\delete\\python')
#os.chdir('E:\\nlp')
os.chdir('D:\\Venkat\\Research\\ReviewSummarization\\Scraping\\results\\mobile-reviews\\')


# In[2]:

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# In[3]:

def clean_sentence(stext):
    stext = stext.split('\n')
    s=''
    for i in stext:
        if len(i.split('-'))==2 and not is_number(i.split('-')[0].split()[-1]):
            s+=''.join(i.split('-')[1:])
        else: s+=' '+i
    stext=s
    stext = stext.split('\n')
    s=''
    for i in stext:
        if len(i.split(':'))==2:s+=''.join(i.split(':')[1:])
        else: s+=' '+i
    stext=s

    stext=re.sub('-.*-','',stext)
    stext = re.sub('[\(0-9]+\)','',stext)
    stext=re.sub('[-{3,}={3,}]',' ',stext)
    stext = re.sub(' {3,}',' ',stext)
    return stext.lstrip()


# In[4]:

'''
part of speech tagging
JJ adjective 'big'
JJR adjective, comparative 'bigger'
JJS adjective, superlative 'biggest'
RB, RBR, RBS Adverbs
tagged = nltk.pos_tag(words)
'''
def eval_words(s):
    words = {}
    for i in nltk.pos_tag(nltk.word_tokenize(s)):
        if i[1] == 'JJ'or i[1]=='JJR' or i[1]=='JJS' or i[1]=='RB' or i[1]=='RBR' or i[1]=='RBS':
            if i[0] in words.keys():words[i[0]]+=1
            else: words[i[0]]=1

    for k,v in words.items():
          if nltk.pos_tag([k])[0][1]=='JJR': words[k]*=1.2
          elif nltk.pos_tag([k])[0][1]=='JJS': words[k]*=1.4
    return words


# In[5]:

def sent_score(s):
    stringTokenizer = StringTokenizer()
    #stok = list(set(sent_tokenize(s)))
    stok = list(set(s.split('\n')))
    weighted_words=eval_words(s)
    d={}
    for i in stok:
        k=0
        swt = [w.lower() for w in nltk.word_tokenize(i)]
        # remove stop words
        #swt = [w for w in swt if not w in stop_words]
        if len(i)>23:
            for w in weighted_words.keys():
                if w in swt and 'i' not in swt and len(swt) <10000 and 'my' not in swt and 'delivery'                    not in swt and 'amazon' not in swt and 'description' not in swt:
                   k+=weighted_words[w]
        d[i]=k
    sorted_stok = sorted(d.items(), key = operator.itemgetter(1),reverse=True)
    return sorted_stok


# In[6]:

## remove similar sentences
def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
    except ValueError:
        return 0
    return ((tfidf * tfidf.T).A)[0,1]


# In[7]:

def summary_review(scored_sentences,cosine_cutoff):
    review=[]
    count = 0
    for i in scored_sentences:
        flag = False
        for k in scored_sentences[:scored_sentences.index(i)]:
            #text1 = re.sub(product_name.lower(),' ',i[0].lower())
            #text2 = re.sub(product_name.lower(),' ',k[0].lower())
            if cosine_sim(i[0],k[0])>cosine_cutoff:
                flag =True
                break
        if not flag: review.append((clean_sentence(i[0]),i[1]))
        #count+=1
        #if count >=15: break
    return review


# In[9]:

def sentimentAnalyser(sentences,pos,neg,result_file):
    pos_reviews=[]
    neg_reviews=[]
    p={}
    n={}
    
    #sid = SentimentIntensityAnalyzer()
    print('\nclassifying sentence sentiment started')
    for sentence in sentences:
        #print(sentence)
        
        blob = TextBlob(sentence[0])
        blob_polarity = blob.sentiment.polarity
        #print(blob_polarity)
        
        #ss = sid.polarity_scores(sentence[0])
        #print(ss)
        
        if blob_polarity > 0.0:
            pos_reviews.append(sentence[0])
            p[sentence[0]]=blob_polarity
        elif blob_polarity < 0.0:
            neg_reviews.append(sentence[0])
            n[sentence[0]]=blob_polarity

        #if ss['pos'] > 0.3:
        #    pos_reviews.append(sentence[0])
        #elif ss['neg'] > 0:
        #    neg_reviews.append(sentence[0])
    print('\nclassifying sentence sentiment done')
    
    print('\nordering positives and negatives')
    sorted_p = sorted(p.items(), key = operator.itemgetter(1),reverse=True)
    sorted_n = sorted(n.items(), key = operator.itemgetter(1),reverse=False)
    #print(sorted_p[:6])
    #print(sorted_n[:6])
    
    print('\nwriting top 6 positive reviews and 6 negative reviews into '+ result_file)
    with open(result_file, 'a', encoding="utf8") as f: 
        f.write('Positive reviews')
        for p in pos_reviews[:6]:
            f.writelines('\n')
            f.writelines(p)
        f.write('\n\nNegative reviews')
        for n in neg_reviews[:6]:
            f.writelines('\n')
            f.writelines(n)
        f.close()
    print('\nwriting to a summary file done')


# In[10]:

if __name__ == '__main__':
    #with open('shoe.txt','r') as f: s=f.read()
    prod_id = 'iphone6'
    ip_filename1 = prod_id+'-reviews.txt'
    ip_filename = prod_id+'.txt'
    op_filename = prod_id+'-summary.txt'
    
    #with open(ip_filename, 'r', encoding="utf8") as f: s=f.read()
    with open(ip_filename, 'r') as f: s=f.read()
    scored_sentences=sent_score(s)
    print("no.of sentences:",len(scored_sentences))
    #for i in scored_sentences: print(i)
    final_review = summary_review(scored_sentences,0.8)
    #for i in final_review: print(i)
    print('\nsentement analysis \n')
    sentimentAnalyser(final_review,0.1,0.1,op_filename)
    print('\nsentement analysis Done \n')


# In[ ]:




# In[ ]:



