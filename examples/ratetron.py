import numpy as np
import pandas as pd
#import scipy.stats as sps
import statsmodels.api as sm
#import statsmodels.tsa.stattools as smt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#import warnings as w


import blpapipd.blpapipd as bdu
#import blpapipd.plot_utilities as pu

diag = False

class rtron_report(list):
    def __init__(self,report_name = 'r_tron.pdf'):
        self.report_name = report_name
        self.pdf_pages = PdfPages(report_name)
        
    def generate_report(self,bbg_session):
        for vi in self:
            print( 'R.Tron: handling %s:%s' % (vi.y, vi.x))
            vi.prep_data(bbg_session)
            print( '         prepping data is done')
            #vi.regressed_backfill()
            #vi.ivolp_df.plot()
            #pu.quick_plot(vi.ivolp_df['splcd'],'%s %s' % (vi.u, vi.ivol[:12]),freq='5AS-JAN')
            #self.pdf_pages.savefig()
            vi.plot_rtron(self.pdf_pages)
            print( '         plots finished')
        self.pdf_pages.close()
    
    
class rtron_item:
    def __init__(self, y, x, bins=[]):
        self.y = y
        self.x = x
        self.bins = bins
        self.y_=[]
        self.x_=[]
    
    def plot_rtron(self,pdf_pages):
        if len(self.bins) == 0:
            cats = pd.cut(self.df[self.x].values,bins=4)
        else:
            cats = pd.cut(self.df[self.x].values,bins=self.bins)

        print( cats)
        print( self.df[self.x].groupby(cats).min())
        print( self.df[self.x].groupby(cats).max())
        reg=self.df.groupby(cats).apply(regress,self.y,[self.x])
        reg['xmin']=self.df[self.x].groupby(cats).min()
        reg['xmax']=self.df[self.x].groupby(cats).max()
        print( reg)

        plt.figure(figsize=(13,10))
        ax = plt.subplot(111)
        self.y_=self.y_[self.y_.index.intersection(self.x_.index)]
        sc = ax.scatter(self.x_[self.y_.index.intersection(self.x_.index)].values,self.y_.values,s=75,c=self.y_.index.year)
        plt.colorbar(sc)
        plt.grid(True)
        xmin,xmax = ax.get_xlim()
        ax.set_xlim([xmin,self.df[self.x].max()*1.15])
        
        ymin,ymax = ax.get_ylim()
        ax.set_ylim([ymin,self.df[self.y].max()*1.15])

        #plt.tight_layout(w_pad=1.,h_pad=1.)

        #plt.text(8,-20,reg.ix[:,0:2].to_string(justify='right',float_format="{:.1f}".format),style='italic',fontsize=10)
        plt.title('%s vs. %s' % (self.y,self.x))

        for idx,r in reg.iterrows():
            x = np.linspace(r['xmin'],r['xmax'],30)
            y = r[self.x]*x + r['Y(0)']
            ax.scatter(x,y,c='orange')
        
        #plt.show()
        pdf_pages.savefig()
        return []
                
    def prep_data(self,s):
        self.y_=bdu.getTimeSeries(s,self.y,'PX_LAST')
        self.x_=bdu.getTimeSeries(s,self.x,'PX_LAST')
        self.y_=self.y_[self.y_.index.intersection(self.x_.index)]
        self.df=pd.DataFrame(np.vstack((self.y_,self.x_[self.y_.index])).T,index=self.y_.index,columns=[self.y,self.x]).dropna()
        print (self.df)
    
def regress(data,yvar,xvars):
    Y = data[yvar]
    X = data[xvars]
    X['Y(0)']=1
    result = sm.OLS(Y,X).fit()
    return result.params
        
def main():
#if True:
    s=bdu.initBBGSession()

    v_r = rtron_report('r_tron.pdf')
    #Energy commodities

    v_r.append(rtron_item(y='USSP2 Curncy',x='USSW2 Curncy'))
    v_r.append(rtron_item(y='USSP5 Curncy',x='USSW5 Curncy'))
    v_r.append(rtron_item(y='USSP7 Curncy',x='USSW7 Curncy'))    
    v_r.append(rtron_item(y='USSP10 Curncy',x='USSW10 Curncy',bins=[1.5,3.5,5.5,7.5,10.75]))
    v_r.append(rtron_item(y='USSP30 Curncy',x='USSW30 Curncy'))
    
    v_r.append(rtron_item(y='USSW210 Curncy',x='USSN0110 Curncy'))
    v_r.append(rtron_item(y='USSW1030 Curncy',x='USSN0130 Curncy'))
    v_r.append(rtron_item(y='USSW530 Curncy',x='USSN0130 Curncy'))
    
    v_r.append(rtron_item(y='USSW210 Curncy',x='FEDL01 Index'))
    
    v_r.generate_report(s)
    
    v_r.pdf_pages.close()
            

if __name__ == "__main__":
    main()
