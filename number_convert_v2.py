# given a pair of integers(x,y), you may perform either of the two operations below in any orders, zero or more times
# 1. (x,y) -> (x+y,y)
# 2. (x,y) -> (x,y+x)
# input four integers a,b,c and d. return yes if it is possible to start with the pair (a,b)
# and end with the pair (c,d) otherwise no.

def oper_a(a):
    b=[]
    b.append(a[1])
    b.append(a[0]+a[1])
    return(b)

def oper_b(a):
    b=[]
    b.append(a[0]+a[1])
    b.append(a[1])
    return(b)

t=[[0,0],[0,1],[1,0],[1,1]]

print('input four numbers one by one')
a=[]
i =0
while (i<4):
    try:
        a.append(int(input()))
        i+=1
    except ValueError:
        print('please enter an integer value')
b=[]
def final_test(a):
    for i in range(len(t)):
        if t[i][0]==0:
            b=oper_a(a[0:2])
        else:
            b=oper_b(a[0:2])
        if t[i][1]==0:
            b=oper_a(b)
        else:
            b=oper_b(b)
        if a[2:]==b:
            return True
            break
    return False
if(final_test(a)):
    print('yes')
else:
    print('no')
        
        
