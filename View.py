from tkinter import *
from pandastable import Table# , TableModel

#View Pandas dataframe in new window
#place this script in your python path directory
#from View import view
#df = TableModel.getSampleData()
#view(df)

def view (df,table_name=None,topmost=True):
    main = Tk()
    main.attributes('-topmost',topmost)
    main.title(table_name)
    f = Frame(main)
    f.pack(fill='both',expand=1)
    pt = Table(f, dataframe=df, showstatusbar=True)
    pt.show()
