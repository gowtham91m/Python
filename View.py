from tkinter import *
from pandastable import Table# , TableModel

#View Pandas dataframe in new window
#place this script in your python path directory
#from View import view
#df = TableModel.getSampleData()
#view(df)

class view():
    def __init__(self,df,table_name=None):
        self.df=df
        self.main = Tk()
        self.main.attributes('-topmost',True)
        self.main.geometry('600x400+200+100')
        self.main.title(table_name)
        f = Frame(self.main)
        f.pack(fill=BOTH,expand=1)
        pt = Table(f, dataframe=df, showstatusbar=True)
        pt.show()
        return
