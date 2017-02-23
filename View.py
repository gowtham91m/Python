from tkinter import *
from pandastable import Table #, TableModel

'''
View Pandas dataframe in new window

Usage
From View import *
View(df) # df is the name of your dataframe
'''

class TestApp(Frame):
    """Basic test frame for the table"""
    def __init__(self,df, parent=None):
        self.df=df
        self.parent = parent
        Frame.__init__(self)
        self.main = self.master
        self.main.geometry('600x400+200+100')
        self.main.title('Table app')
        f = Frame(self.main)
        f.pack(fill=BOTH,expand=1)
        #df = TableModel.getSampleData()
        self.table = pt = Table(f, dataframe=self.df,
                                showtoolbar=True, showstatusbar=True)
        pt.show()
        return

        
def View(df):
    app = TestApp(df)#launch the app
    app.mainloop()

