slist=list(input('enter the string '))

def get_distinct(original_list):
    distinct_list = []
    for each in original_list:
        if each not in distinct_list:
            distinct_list.append(each)
    return distinct_list
count = len(get_distinct(slist))

blist=[]
for i in range(len(slist)-1):
        k=list(range(i+2))
        n=0
        while n <= (len(slist)-len(k)):
            n+=1
            plist=slist[k[0]:k[-1]+1]
            k=list(k[j]+1 for j in range(len(k)))
            if (plist==plist[::-1]):
                if plist not in blist:
                    blist.append(plist)
##                print('palindrome',plist)
##print(blist)
count = count+len(blist)
print(count)
        
    
