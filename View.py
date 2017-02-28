from pandastable import Table, TableModel
from tkinter import *

'''
using:
from pandastable import TableModel
from View import view
df = TableModel.getSampleData()
view(df)
'''

class view():
    def __init__(self,df):
        self.tk=Tk()
        self.tk.geometry('600x400+200+100')
        self.tk.title('Table View')
        self.f=Frame(self.tk)
        self.f.pack(fill=BOTH,expand=1)
        pt=Table(self.f,dataframe=df,showtoolbar=True, showstatusbar=True)
        pt.show()
