import numpy as np
import pandas as pd
#import scipy.stats as sps
import statsmodels.api as sm
import statsmodels.tsa.stattools as smt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings as w

import blpapipd.plot_voltron as pv

import blpapipd.plot_utilities as pu
import blpapipd.blpapipd as bdu

diag = False

class vtron_report(list):
    def __init__(self,report_name = 'vol_tron.pdf'):
        self.report_name = report_name
        self.pdf_pages = PdfPages(report_name)
        
    def generate_report(self,bbg_session):
        # iterate over vtron_items
        for vi in self:
            print('VolTron: handling %s:%s' % (vi.u, vi.ivol))
            vi.prep_data(bbg_session)
            print( '         prepping data is done')
            vi.regressed_backfill()
            #print( '         regressed backfill done')
            #vi.ivolp_df.plot()
            pu.quick_plot(vi.ivolp_df['splcd'],'%s %s' % (vi.u, vi.ivol[:12]),freq='5AS-JAN')
            self.pdf_pages.savefig()
            vi.plot_voltron(self.pdf_pages)
            print( '         plots finished')
        self.pdf_pages.close()
    
    
class vtron_item:
    def __init__(self, u, fld, u_sigma, ivol, ivol_sigma, ivol_is_subticker=True):
        self.u = u
        self.fld = fld
        self.u_sigma = u_sigma
        self.ivol = ivol
        self.ivol_is_subticker = ivol_is_subticker
        self.ivol_sigma = ivol_sigma
        self.OLS_r = None
        self.ADF_r = ()
        self.u_gaps = pd.DataFrame()
        self.ivol_gaps = pd.DataFrame()
        self.ivolp_df  = pd.DataFrame()
    
    def plot_voltron(self,pdf_pages):
        pv.plot_voltron(self,pdf_pages)
            
    def prep_data(self,bbg_session,plot=False):
        self.u_df,self.u_gaps       = get_clean_ts(bbg_session,self.u,self.fld, False, self.u_sigma)
        if self.ivol_is_subticker:
            self.ivol_df,self.ivol_gaps = get_clean_ts(bbg_session,self.u,self.ivol, False, self.ivol_sigma)
        else:
            self.ivol_df,self.ivol_gaps = get_clean_ts(bbg_session,self.ivol,self.fld, False, self.ivol_sigma)
        
    def regressed_backfill(self):
        global diag
        #make sure the data is clean, used get_clean_ts so should be ok
            
        ivol_orig_ts = self.ivol_df['orig']            
        ivol_ts = self.ivol_df['clnd']
                
        u_ts    = self.u_df['clnd']
        u_ts    = u_ts[u_ts.index.intersection(ivol_ts.index)]
            
        if len(ivol_ts.index) >= len(u_ts):
            #shorten ivol history by 252
            ivol_ts = ivol_ts[ivol_ts[252:].index]

        tmp_df=pd.DataFrame(ivol_ts)
        tmp_df['undrlyng']=u_ts
        tmp_df=tmp_df.ffill()
        #tmp_df.to_csv('tmp_df.csv')
                
        #produce independent variables for relevant time period
        # 1m rvol, 3m rvol, 3m3m rvol, 1y rvol
        X_df=pd.DataFrame(u_ts)
        X_df['rvol_1m']    = u_ts.pct_change().rolling(21).std()*1585
        X_df['rvol_3m']    = u_ts.pct_change().rolling(63).std()*1585
        X_df['rvol_3m_3m'] = X_df['rvol_3m'].shift(63)
        X_df['rvol_1y']    = u_ts.pct_change().rolling(252).std()*1585
            
        #can't have any missing dates, because these propagate into rolling_std.
        #make sure that all dates are represented after u_ts has been fwd filled.
        X2_df = pd.DataFrame(X_df['clnd'].ffill(),columns=['clnd'])
        X2_df['rvol_1m']    = X2_df['clnd'].pct_change().rolling(21).std()*1585    
        X2_df['rvol_3m']    = X2_df['clnd'].pct_change().rolling(63).std()*1585    
        X2_df['rvol_3m_3m'] = X2_df['rvol_3m'].shift(63)
        X2_df['rvol_1y']    = X2_df['clnd'].pct_change().rolling(252).std()*1585    
            
            
        X2_df['y'] = ivol_ts
        X2_df.ffill(inplace=True)   #at this point, only ivol should have NaNs due to missing dates
        X2_df.dropna(inplace=True)  #this should drop rvol data before ivol data exists (checked that ivol is smaller and more recent than u_ts)
            
        #align series
        
        #   rvol_1m    = rvol_1m[ivol_ts.index]
        #   rvol_3m    = rvol_3m[ivol_ts.index]
        #   rvol_3m_3m = rvol_3m_3m[ivol_ts.index]
        #   rvol_1y    = rvol_1y[ivol_ts.index]
            
        # run regression, find significant factors
        #   X=np.column_stack( (rvol_1m.values, rvol_3m.values,rvol_3m_3m.values,rvol_1y.values) )
        #   X = sm.add_constant(X)
        #   X_df = pd.DataFrame(X)
        
        #X2_df.to_csv('X_df.csv')
            
        if len(X2_df) - len(ivol_ts) > 3:
            raise('len(X) not equal to len(ivol_ts.values)')
            
              
        model = sm.OLS(X2_df['y'].values, X2_df[['clnd','rvol_1m','rvol_3m','rvol_3m_3m','rvol_1y']].values)
        results = model.fit()
        if diag:
            print('OLS Regression results:\n')
            print(results.summary())
        
                
        r=smt.adfuller(results.resid)
        if diag:
            print ('ADF test on residuals:\n')
            print( r)

        # extend ivol_df using regression results
        # calc longer
        u_ts = self.u_df['clnd'].ffill()
            
        Xp_df=pd.DataFrame(u_ts)
        Xp_df['rvol_1m']    = u_ts.pct_change().rolling(21).std()*1585    
        Xp_df['rvol_3m']    = u_ts.pct_change().rolling(63).std()*1585    
        Xp_df['rvol_3m_3m'] = Xp_df['rvol_3m'].shift(63)
        Xp_df['rvol_1y']    = u_ts.pct_change().rolling(252).std()*1585
            
        #Xp_df.to_csv('Xp_df.csv')
            
        # backfill to extent of underlying
        # matrix of data, to be dot multiplied by regression coeffs
        if diag:
            print (np.dot(Xp_df[['clnd','rvol_1m','rvol_3m','rvol_3m_3m','rvol_1y']].values,results.params.T))
            
        # start w rgrsd data, as it sets the time axis of the time-series    
        self.ivolp_df=pd.DataFrame(np.dot(Xp_df[['clnd','rvol_1m','rvol_3m','rvol_3m_3m','rvol_1y']].values,results.params.T),index=Xp_df.index,columns=['rgrsd'])
        self.ivolp_df['orig']=ivol_orig_ts
        self.ivolp_df['clnd']=ivol_ts
        self.ivolp_df['splcd']=pd.Series(self.ivolp_df['rgrsd'].values,index=self.ivolp_df.index)
        self.ivolp_df['splcd'][np.intersect1d(ivol_ts.index,self.ivolp_df.index)] = ivol_ts[np.intersect1d(ivol_ts.index,self.ivolp_df.index)]
        self.ivolp_df = self.ivolp_df[['orig','clnd','rgrsd','splcd']] #re-org the columns for later plotting
        
        self.OLS_r = results
        self.ADF_r = r

def condition_ts(ts,diff_type='pct_change',sigma=5,periods=[1],cure_type='drop',recur_depth=0):
    global diag
    
    ts_r = ts
    outliers_df = pd.DataFrame()
    
    #identify time gaps, return cure
    i=ts.index
    isr=i.to_series()
    #print all periods where timeseries 'jumps' by more than 4 days
    
    bool_idx=isr.diff() > np.timedelta64(4,'D')
    time_gaps_df=pd.DataFrame(isr.diff()[bool_idx],columns=['time gaps'])
    if diag:
        print( time_gaps_df)
        print( '\n')
   
    #store time axis cure
        
    #identify large, anomalous diffs
    for p in periods:
        f=getattr(ts_r,diff_type)
        if callable(f):
            r = np.log(1+f(periods=p))
            z = (r-r.mean())/r.std()
            
            #z.hist(bins=50)
            #plt.show()
            
            #z[~np.isnan(r)] = sps.zscore(r[~np.isnan(r)],axis=0)
            #z3 = (r-pd.rolling_mean(r,20))/pd.rolling_std(r,20)
            #z4 = (ts_r-pd.rolling_mean(ts_r,20))/pd.rolling_std(ts_r,20)


            #create boolean vector that rejects the data that is n period wide
            b =abs(z)>sigma
            #b3=abs(z3)>sigma
            #b4=abs(z4)>sigma

            # example for a period of 3 days wide
            # b   = [ 0 0 0 1 0 0 1 0 1]
            # b_1 = [ 0 0 1 0 0 1 0 1 0]
            # b_2 = [ 0 1 0 0 1 0 1 0 0]  
            # b_r = or(b,b_1,b_2)
            # and then ts.tail(-1)[not b]
#            b_shftd=b
#            for i in np.arange(p-1):
#                b_shftd = np.roll(b_shftd,-1)
#                b_shftd = np.append(np.delete(b_shftd,len(b_shftd)-1),[False])
#                b=np.logical_or(b,b_shftd)

            outliers_df=pd.DataFrame(ts_r.tail(-p)[b],columns=['%d,%.2f sigma outliers' % (p,sigma)])
            if diag:
                print( outliers_df)
            # setting to NaN approach doesn't work.  Can't compute diff in a recursive outlier search
            if len(ts_r) == len(b):
                tmp_df = pd.DataFrame(ts_r,columns=['ts_r'])
                tmp_df['b']=b
                #tmp_df['b2']=b2
                #tmp_df['b3']=b3
                #tmp_df['b4']=b4
                tmp_df['z']=z
                #tmp_df['abs(z2)']=abs(z2)
                #tmp_df['abs(z3)']=abs(z3)
                #tmp_df['abs(z4)']=abs(z4)
                ts_r=ts_r[np.logical_not(b)].copy(deep=True)
                if diag:                                
                    print( tmp_df)
            else:
                print( 'ts_r not equal to boolean index'   )
    
            #check to see if there are any moves larger than n-sigma left
            f = getattr(ts_r,diff_type)
            r = np.log(1+f(periods=p))
            z = (r-r.mean())/r.std()
    
            # hold off on recursion for now... problem if there are two-three downspikes that are less than 3-sigma, followed by a return to normal level of 3-sigma or more.       
            if len(ts_r[abs(z)>sigma]) > 0 and recur_depth<3:
                recur_depth+=1
                ts_r, discard, tmp = condition_ts(ts_r,diff_type,sigma,periods,cure_type,recur_depth)
                outliers_df.append(tmp)
            
    return ts_r, time_gaps_df, outliers_df
 

        
def get_clean_ts(s,tkr,fld,plot=False,sigma=5):
    d = bdu.getTimeSeries(s,tkr,fld)
    d_cln,time_gaps_df, outliers_df = condition_ts(d,sigma=sigma)
    r=pd.DataFrame(d,columns=['orig'])
    r['clnd']=d_cln
    # find latest episode of more than 10days of missing data
    latest_time_gap = time_gaps_df[time_gaps_df['time gaps'] > np.timedelta64(15,'D')].tail(1).index
#    if not latest_time_gap.empty:
    df=pd.DataFrame(latest_time_gap)
    if not df.empty:
        #print latest_time_gap
        #print len(r)
        r = r[latest_time_gap.format()[0]:]
        #print len(r)
        
    if plot:
        r.plot(title=tkr+' '+fld)
        plt.show()
    return r, time_gaps_df

#def main():
if True:
    w.simplefilter('ignore')
    s=bdu.initBBGSession()

    v_r = vtron_report('vol_tron.pdf')
    #Energy commodities

    v_r.append(vtron_item(u='CL1 Comdty',fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=7))
    v_r.append(vtron_item(u='CL1 Comdty',fld='PX_LAST',u_sigma=7,   ivol='24MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=7))

    v_r.append(vtron_item(u='CO1 Comdty',fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=7))    
#    v_r.append(vtron_item(u='NG1 Comdty',fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=7))    
                
    #softs        
    v_r.append(vtron_item(u='S 1 Comdty',fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=7))    
    v_r.append(vtron_item(u='CT1 Comdty',fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=7))
    v_r.append(vtron_item(u='CC1 Comdty',fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=7))
            
    #LME metals
    #v_r.append(vtron_item(u='LP1 Comdty',fld='PX_LAST',u_sigma=5,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))

    #COMEX metals    
    v_r.append(vtron_item(u='GC1 Comdty',fld='PX_LAST',u_sigma=5,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))
    v_r.append(vtron_item(u='SI1 Comdty',fld='PX_LAST',u_sigma=5,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))
    v_r.append(vtron_item(u='HG1 Comdty',fld='PX_LAST',u_sigma=5,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))

    #Indexes            
    v_r.append(vtron_item(u='NKY Index' ,fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))
    v_r.append(vtron_item(u='UKX Index' ,fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))
    #v_r.append(vtron_item(u='SX5E Index' ,fld='PX_LAST',u_sigma=7,  ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))
    v_r.append(vtron_item(u='SPX Index' ,fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))
    v_r.append(vtron_item(u='RTY Index' ,fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))

    #Select ETFs
    v_r.append(vtron_item(u='LQD Index' ,fld='PX_LAST',u_sigma=7,   ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))
    v_r.append(vtron_item(u='HYG Equity' ,fld='PX_LAST',u_sigma=7,  ivol='12MTH_IMPVOL_100.0%MNY_DF', ivol_sigma=5))                
    
    #FX
    v_r.append(vtron_item(u='EURUSD Curncy' ,fld='PX_LAST',u_sigma=7,   ivol='EURUSDV1Y Curncy', ivol_sigma=5, ivol_is_subticker=False))
    v_r.append(vtron_item(u='USDJPY Curncy' ,fld='PX_LAST',u_sigma=7,   ivol='USDJPYV1Y Curncy', ivol_sigma=5, ivol_is_subticker=False))
    v_r.append(vtron_item(u='GBPUSD Curncy' ,fld='PX_LAST',u_sigma=7,   ivol='GBPUSDV1Y Curncy', ivol_sigma=5, ivol_is_subticker=False))
    #v_r.append(vtron_item(u='USDRUB Curncy' ,fld='PX_LAST',u_sigma=7,   ivol='USDRUBV1Y Curncy', ivol_sigma=5, ivol_is_subticker=False))
    #    v_r.append(vtron_item(u='CHFUSD Curncy' ,fld='PX_LAST',u_sigma=7,   ivol='CHFUSDV1Y Curncy', ivol_sigma=5, ivol_is_subticker=False))
    #    v_r.append(vtron_item(u='CADUSD Curncy' ,fld='PX_LAST',u_sigma=7,   ivol='CADUSDV1Y Curncy', ivol_sigma=5, ivol_is_subticker=False))
    
    #rates
    v_r.append(vtron_item(u='USSW10 Curncy' ,fld='PX_LAST',u_sigma=7, ivol='USSN0110 Curncy', ivol_sigma=5, ivol_is_subticker=False))
        
    v_r.generate_report(s)

#    ct1_1y_ivol = get_clean_ts(s,'CT1 Comdty','12MTH_IMPVOL_100.0%MNY_DF')
#    ct1         = get_clean_ts(s,'CT1 Comdty','PX_LAST',sigma=7)
#    gc1_1y_ivol = get_clean_ts(s,'GC1 Comdty','12MTH_IMPVOL_100.0%MNY_DF')
#    gc1         = get_clean_ts(s,'GC1 Comdty','PX_LAST',sigma=7)

#    rty         = get_clean_ts(s,'RTY Index','PX_LAST',sigma=7)
#    rty_1y_ivol = get_clean_ts(s,'RTY Index','12MTH_IMPVOL_100.0%MNY_DF',sigma=5)

#    cc1_1y_ivol = get_clean_ts(s,'CC1 Comdty','12MTH_IMPVOL_100.0%MNY_DF',sigma=8)
#    pu.quick_plot(cc1_1y_ivol['clnd'],'Cocoa  1yr atm i.vol')

#    print [x for x in vtron_list]
            
    if False:
        cc1_1y_ivol = get_clean_ts(s,'CC1 Comdty','12MTH_IMPVOL_100.0%MNY_DF')
        cl1_1y_ivol = get_clean_ts(s,'CL1 Comdty','12MTH_IMPVOL_100.0%MNY_DF')
        co1_1y_ivol = get_clean_ts(s,'CO1 Comdty','12MTH_IMPVOL_100.0%MNY_DF')
        ng1_1y_ivol = get_clean_ts(s,'NG1 Comdty','12MTH_IMPVOL_100.0%MNY_DF',sigma=6)
        spx_1y_ivol = get_clean_ts(s,'SPX Index','12MTH_IMPVOL_100.0%MNY_DF',sigma=7)
        rty_1y_ivol = get_clean_ts(s,'RTY Index','12MTH_IMPVOL_100.0%MNY_DF',sigma=7)
        indu_1y_ivol = get_clean_ts(s,'INDU Index','12MTH_IMPVOL_100.0%MNY_DF',sigma=7)


    if False:
        pu.quick_plot(ct1_1y_ivol,'Cotton 1yr atm i.vol')
        pu.quick_plot(cc1_1y_ivol,'Cocoa  1yr atm i.vol')
        pu.quick_plot(gc1_1y_ivol,'Gold 1yr atm i.vol')
        pu.quick_plot(cl1_1y_ivol,'WTI 1yr atm i.vol')
        pu.quick_plot(co1_1y_ivol,'Brent 1yr atm i.vol')
        pu.quick_plot(spx_1y_ivol,'SPX 1yr atm i.vol')


        pu.quick_plot(indu_1y_ivol,'INDU 1yr atm i.vol')
    
    
 #   r1=regressed_backfill(ct1_1y_ivol,ct1)
 #   r1['u_name']='CT1 Comdty'
 #   r1['ivol_name']='CT1 12mo ivol'
 #   r2=regressed_backfill(gc1_1y_ivol,gc1)
 #   r3=regressed_backfill(rty_1y_ivol,rty)
 #   pu.quick_plot(rty_1y_ivol['clnd'],'RTY 1yr atm i.vol')
    
 #for loop over commodity, ivol pairs.  needs to specify sigma tolerance as well.
       # get clean ts  underlying
       # get clean ts  i.vol
       
       # run regressed backfill
       
       # plot_voltron   passed a file handle as well.
       
    #keep dictionary of results and tabulate summary
     # avg, min, max, argmin, argmax, std, %ile, z-score, 3-sigma
     # voltron alert scale
     #    6mo underlying correlation
     #    6mo or 1yr i.vol
     #    i.vol/r.vol
     
    #pv.plot_voltron(r1)
    

#if __name__ == "__main__":
#    main()
