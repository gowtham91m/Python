from tkinter import *
from pandastable import Table #, TableModel

'''
View Pandas dataframe in new window
place this script in your python path directory
from View import view
Usage
From View import *
df = TableModel.getSampleData()
View(df)
'''

class TestApp():
    """Basic test frame for the table"""
    def __init__(self,df, master=None):
        self.df=df
        self.main = master
        self.main.attributes('-topmost',True)
        self.main.geometry('600x400+200+100')
        self.main.title('Table app')
        f = Frame(self.main)
        f.pack(fill=BOTH,expand=1)
        pt = Table(f, dataframe=df, showstatusbar=True)
        pt.show()
        return
        
def view(df):
        root = Tk()
        app = TestApp(df,root)
