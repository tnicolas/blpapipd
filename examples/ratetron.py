import numpy as np
import pandas as pd
import re as re
#import scipy.stats as sps
import statsmodels.api as sm
#import statsmodels.tsa.stattools as smt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#import warnings as w



import blpapipd.blpapipd as bdu
import blpapipd.plot_utilities as pu

diag = False

class rtron_report(list):
    def __init__(self,report_name = 'r_tron.pdf'):
        self.report_name = report_name
        self.pdf_pages = PdfPages(report_name)
        
    def generate_report(self,bbg_session):
        for vi in self:
            print( 'R.Tron: handling %s:%s:%s from %s to %s' % (vi.x, vi.y, vi.z, vi.st_dt, vi.ed_dt))
            vi.prep_data(bbg_session)
            print( '         prepping data is done')
            #vi.regressed_backfill()
            #vi.ivolp_df.plot()
            #pu.quick_plot(vi.ivolp_df['splcd'],'%s %s' % (vi.u, vi.ivol[:12]),freq='5AS-JAN')
            #self.pdf_pages.savefig()
            vi.plot_rtron(self.pdf_pages)
            print( '         plots finished')
        #self.pdf_pages.close()
    
    
class rtron_item:
    def __init__(self, row):
        self.x = row['X']
        self.y = row['Y']
        self.z = row['Z']
        if pd.notna(row.bins):
            self.bins = [float(ele) for ele in row['bins'][1:-1].split(',')]
        else:
            self.bins = []
        self.fn   = row['pu func']
        self.st_dt = row['st_dt']
        if pd.isna(self.st_dt):
            self.st_dt = []
        self.ed_dt = row['end_dt']
        if pd.isna(self.ed_dt):
            self.ed_dt = []
        # add an excerpting feature... takes a list of pair tuples, of periods to exclude from the data, passed in xclud column
        self.y_=[]
        self.x_=[]
        self.z_=[]
        #print(self)
    
    def plot_rtron(self,pdf_pages):
        
        #make this a binned, colormapped regression function
        if self.fn == 'scatter':
            if len(self.bins)==0:
                cats = pd.cut(self.df[self.x].values,bins=4)
            else:
                cats = pd.cut(self.df[self.x].values,bins=self.bins)

            #print( cats)
            #print( self.df[self.x].groupby(cats).min())
            #print( self.df[self.x].groupby(cats).max())
            
            reg=self.df.groupby(cats).apply(regress,self.y,[self.x])
            reg['xmin']=self.df[self.x].groupby(cats).min()
            reg['xmax']=self.df[self.x].groupby(cats).max()
            #print( reg)

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
            
            plt.tight_layout(w_pad=1.,h_pad=1.)
            latest_y=self.y_.tail(1)
            latest_x=self.x_.tail(1)
            plt.plot(latest_x[0],latest_y[0],'ro',markersize=10)
            
            #plt.text(8,-20,reg.ix[:,0:2].to_string(justify='right',float_format="{:.1f}".format),style='italic',fontsize=10)
            plt.title('%s vs. %s' % (self.y,self.x))
            
            for idx,r in reg.iterrows():
                x = np.linspace(r['xmin'],r['xmax'],30)
                y = r[self.x]*x + r['Y(0)']
                ax.scatter(x,y,c='orange')
        
            #plt.show()
            pdf_pages.savefig(ax.get_figure())
            return []
        elif self.fn == 'quick_plot':
            fig=pu.quick_plot(self.x_,self.x,ln_std=False,bands=True,freq='AS-JAN',bottomLogo=False)
            pdf_pages.savefig(fig)
        elif self.fn == 'two_plot':
            fig=pu.two_plot(self.x_, self.y_, self.x +" & "+self.y, ts1_name=self.x, ts2_name=self.y,freq='AS-JAN',secondary_y=True,invert_ts1=False,invert_ts2=False,bottomLogo=False)
            pdf_pages.savefig(fig)
        elif self.fn == 'curve_plot':
            fig=pu.quick_plot(self.df[self.y]-self.df[self.x],self.y+'-'+self.x,ln_std=False)
            pdf_pages.savefig(fig)
        elif self.fn == 'fly_plot':
            fig=pu.quick_plot(2*self.df[self.y]-self.df[self.x]-self.df[self.z],'2x'+self.y+'-'+self.x+'-'+self.z,ln_std=False)
            pdf_pages.savefig(fig)
        elif self.fn == 'px_quick_plot':
             fig=pu.px_quick_plot(self.x_,self.x,ln_std=False,bands=True,freq='AS-JAN',bottomLogo=False)
             fig.show()
             #pdf_pages.savefig(fig)
        elif self.fn == 'annual_seasonal':
             fig=pu.seasonal_plot(self.x_,self.x)
             fig.show()
        else:
            print('don\'t recognize the plot')
        
        return []
    
    def prep_data(self,s):
        if pd.notna(self.y) and pd.notna(self.z):
            self.x_=bdu.getTimeSeries(s,self.x,'PX_LAST',self.st_dt,self.ed_dt)
            self.y_=bdu.getTimeSeries(s,self.y,'PX_LAST',self.st_dt,self.ed_dt)
            self.z_=bdu.getTimeSeries(s,self.z,'PX_LAST',self.st_dt,self.ed_dt)
            
            self.y_=self.y_[self.y_.index.intersection(self.x_.index)]
            self.z_=self.z_[self.z_.index.intersection(self.y_.index)]
            self.df=pd.DataFrame(np.vstack([self.x_[self.z_.index],self.y_[self.z_.index],self.z_]).T,index=self.z_.index,columns=[self.x,self.y,self.z]).dropna()
            
        elif pd.notna(self.y) and pd.isna(self.z):
            self.x_=bdu.getTimeSeries(s,self.x,'PX_LAST').dropna()
            self.y_=bdu.getTimeSeries(s,self.y,'PX_LAST').dropna()
            self.y_=self.y_[self.y_.index.intersection(self.x_.index)]
            self.df=pd.DataFrame(np.vstack([self.x_[self.y_.index],self.y_]).T,index=self.y_.index,columns=[self.x,self.y]).dropna()
            
        elif pd.notna(self.x):
            self.x_=bdu.getTimeSeries(s,self.x,'PX_LAST',self.st_dt,self.ed_dt)
            #print(self.x_)
            self.df=pd.DataFrame(self.x_,index=self.x_.index,columns=[self.x]).dropna()
        else:
            print('error')
            return(0)
        
        #print (self.df)
    
def regress(data,yvar,xvars):
    Y = data[yvar]
    X = data[xvars]
    X['Y(0)']=1
    if len(X):
        result = sm.OLS(Y,X).fit()
        return result.params
        
#take a file name that lists single and double r_tron items
#the output file takes its name from input file
def main():
#if True:
    inp = input('enter excel report list: ')
    print(inp)

    
    x_y_z_items=pd.read_csv(inp)
    print(x_y_z_items)
    
    s=bdu.initBBGSession()
    v_r = rtron_report(re.sub('csv','pdf',inp))

    #print(x_y_items)
    for index, row in x_y_z_items.iterrows():
        v_r.append(rtron_item(row))
        
    v_r.generate_report(s)
    
    v_r.pdf_pages.close()
            

if __name__ == "__main__":
    main()
