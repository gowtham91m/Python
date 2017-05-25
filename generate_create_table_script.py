## Oracle

create = 'create table instacart_train('
for i in train.columns:
    if train[i].dtypes == 'object':
        create = create +' '+str(i)+' varchar2('+str(train[i].map(len).max()+2)+'),\n'
    else:
        create = create +' '+str(i)+' int,\n'
create=create[:-2]+')'
print(create)


## postgreSQL

create = 'create table instacart_train('
for i in train.columns:
    if train[i].dtypes == 'object':
        create = create +' '+str(i)+' text,\n'
    else:
        create = create +' '+str(i)+' int,\n'
create=create[:-2]+')'
print(create)
