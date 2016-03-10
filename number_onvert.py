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

n=[1,2]
##print(oper_a(n))
##print(oper_b(n))

print('input four numbers one by one')
a=[]
i =0
while (i<4):
    try:
        a.append(int(input()))
        i+=1
    except ValueError:
        print('please enter an integer value')

def final_test(a):
    t=[]
    t=oper_a(a[0:2])
    print(t)
    t=oper_a(t)
    print(t)
    if a[2:]==t:
        return ('Yes')
    else:
        t=oper_a(a[0:2])
        t=oper_b(t)
        print(a)
        print(a[2:])
        if a[2:]==t:
            return ('Yes')
        else:
            t=oper_b(a[0:2])
            t=oper_a(t)
            if a[2:]==t:
                return ('Yes')
            else:
                t=oper_b(a[0:2])
                t=oper_b(t)
                if a[2:]==t:
                    return ('Yes')
                else:
                    return ('No')
print(final_test(a))

    
