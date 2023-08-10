import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import pandas as pd
import plotly.express as px
#import colormaps as cm
import seaborn as sns



from matplotlib.ticker import MaxNLocator
#import Image
from matplotlib.font_manager import FontProperties

def regr_plot(x, y, ttle, step=0.1, xlabel="", ylabel="", freq_counter=False, order=1,bottomLogo=False):
    # run some regressions to gain a sense of relationship, levels and diffs

    #x_= np.arange(x.min(),x.max(),step)
    z = np.polyfit(x,y,2)
    p = np.poly1d(z)

    fig=plt.figure(figsize=(12,10))
    plt.title(ttle)
 
    ax=fig.add_subplot(111)
    if freq_counter:
        ax.xaxis.set_major_locator(MaxNLocator(symmetric=True))
        
    #y_recent_y = []; x_recent_y = []; y_recent_q=[]; x_recent_q =[]
 
    #x_recent_y = x.tail(int(0.20*x.shape[0])) # current year
    #y_recent_y = y[x_recent_y.index]  
    
#    if x.shape[0] > 252:
#        x_recent_y = x.tail(252) # current year
#        y_recent_y = y[x_recent_y.index]
#                
#        x_recent_q = x.tail(63) # current quarter
#        y_recent_q = y[x_recent_q.index]
#    if x.shape[0] > 63:
#        x_recent_q = x.tail(63) # current quarter
#        y_recent_q = y[x_recent_q.index]
#    elif x.shape[0] > 12:
#        x_recent_q = x.tail(12) # current quarter
#        y_recent_q = y[x_recent_q.index]

                
    sc=ax.scatter(x.values,y.values,s=75,c=y.index.year,cmap=sns.color_palette('viridis'))
    cbar=plt.colorbar(sc)
    cbar.ax.get_yaxis().set_ticks([])
    
    ax.scatter(x.tail(1).values,y.tail(1).values,c='red',s=85)
    ax.plot(x.values,p(x.values),'g-')
    
    #need to plot latest click
    #need to plot scatter line
    
    xmin,xmax = ax.get_xlim()
    ax.set_xlim([xmin,x.max()*1.15])
        
    ymin,ymax = ax.get_ylim()
    ax.set_ylim([ymin,y.max()*1.15])
#    plt.setp(lines[0],'label','regr')
#    plt.setp(lines[1],'label','data')
#    plt.setp(lines[2],'label','recent data')
    #xmin,xmax = ax.get_xlim()
    #ax.set_xlim([x.min()*1.15,x.max()*1.15])
            
    #ax.set_xlabel('%s  [ m:%.3f, b:%.3f    1/m: %.3f ]' % (xlabel,m,b,1./m),fontsize=12,style='italic')
    ax.set_xlabel(xlabel,fontsize=12,style='italic')
    if freq_counter:
        ax.text(0.25,0.75,'N = %d' % sum(np.logical_and(x<0,y>0)),transform=ax.transAxes )
        ax.text(0.75,0.75,'N = %d' % sum(np.logical_and(x>0,y>0)),transform=ax.transAxes )

    plt.grid()

    plt.ylabel(ylabel)
    plt.tight_layout(w_pad=0.5,h_pad=0.5)
    plt.legend(loc='upper right',prop={'size':10})
    plot_logo(ax,bottomLogo)
    plt.show()

    return ax

def two_plot_df(df,freq='AS-JAN',ttle="",secondary_y=True,invert_ts1=False,invert_ts2=False,bottomLogo=False):
    # run some regressions to gain a sense of relationship, levels and diffs

    c = df.columns
    ts1_name=c[0]
    ts1=pd.Series(index=df.index,data=df[ts1_name].values)    
    
    ts2_name=c[1]
    ts2=pd.Series(index=df.index,data=df[ts2_name].values)    

    if ttle == "":
        ttle = ts1_name + " & " + ts2_name
    
            
    ax = two_plot(ts1, ts2, ttle, ts1_name, ts2_name, freq, secondary_y,invert_ts1,invert_ts2,bottomLogo) 
    return ax

def two_plot(ts1, ts2, ttle, ts1_name="", ts2_name="",freq='AS-JAN',secondary_y=True,invert_ts1=False,invert_ts2=False,bottomLogo=False):
    # run some regressions to gain a sense of relationship, levels and diffs

    s_dt=ts1.index[0]
    e_dt=ts1.index[-1]
    xtks=pd.date_range(start=s_dt,end=e_dt,freq=freq)

    #if invert_ts2:
    #    ts2=-1*ts2
    if invert_ts1:
        ts1_name = ts1_name + ' (inverted)'
    
    if invert_ts2:
        ts2_name = ts2_name + ' (inverted)'

    df=pd.DataFrame(ts1.values,index=ts1.index,columns=[ts1_name])

    df[ts2_name]=ts2
    df['%s mean' % (ts1_name)] = np.mean(ts1)
    df['%s mean' % (ts2_name)] = np.mean(ts2)
    
    if secondary_y:        
        ax=df.plot(figsize=(12,10),xticks=xtks.to_pydatetime(),color=['k', '#ff7f00','k','#ff7f00'],lw=2,secondary_y=[ts2_name, '%s mean' % ts2_name],grid=True)
        ymin,ymax = ax.right_ax.get_ylim()
        ax.right_ax.set_ylim([ymin,ts2.max()*1.15])
    else:
        ax=df.plot(figsize=(12,10),xticks=xtks.to_pydatetime(),color=['k', '#ff7f00','k','#ff7f00'],lw=2, grid=True)
    

    
    ax=plt.gca()
    if invert_ts1:
        ax.left_ax.invert_yaxis()
    if invert_ts2:
        ax.right_ax.invert_yaxis()
        
    fig=ax.get_figure()
    fig.set_facecolor('whitesmoke')
    
    plt.title(ttle)

    plt.legend(loc='upper right',prop={'size':10})
    plt.tight_layout(w_pad=0.5,h_pad=0.5)
    #plot_logo(ax,bottomLogo)
    plt.show()

    return fig



def quick_plot(ts,c_name,ln_std=True,bands=True,freq='AS-JAN',bottomLogo=False):
    month_d={1:'Jan', 2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

    s_dt=ts.index[0]
    e_dt=ts.index[-1]
    extent = min(len(ts),252)
    
    end_dt=e_dt+dt.timedelta(days=90)
    xtks=pd.date_range(start=s_dt,end=end_dt,freq=freq)
    df=pd.DataFrame(ts.values,index=ts.index,columns=[c_name])
    df['mean']= ts.mean()
#    df['max'] = ts.max()
#    df['min'] = ts.min()
    last=df[c_name].tail(1)

    pctile = 100*float(np.sum(ts < ts[-1]))/ts.shape[0]

    lvl_std = np.std(df[c_name])
    std_1y  = np.std(df[c_name].tail(extent))
    df['+-1 zscore %0.1f %0.1f' % (ts.mean()-lvl_std, ts.mean()+lvl_std)] = ts.mean()-lvl_std
    df[''] = ts.mean()+lvl_std
    df['-0.5 zscore %0.1f' % (ts.mean()-0.5*lvl_std)] = ts.mean()-0.5*lvl_std

    ax=df.plot(figsize=(12,10),xticks=xtks.to_pydatetime(),color=['k','#ff7f00','.6','.6','.7'],lw=2)
    ymin,ymax = ax.get_ylim()
    ax.set_ylim([ts.min()-abs(ts.mean()*.1),ts.max()+abs(ts.mean()*.1)])
    fig=ax.get_figure()
    fig.set_facecolor('whitesmoke')
    #ax.set_axis_bgcolor('gray')
    
    
#    df_stub=pd.DataFrame(last[0]*np.exp(ret_std),index=pd.date_range(e_dt,end_dt),columns={'+-1 rstd'})
#    df_stub.plot(ax=ax)

    #should check that data is given in daily frequency
    if(ln_std):
        ret_std = np.std(np.log((df[c_name].pct_change()+1)))*np.sqrt(252)
    else:
        ret_std = np.std(df[c_name].diff())*np.sqrt(252)

    largest_drop_in_1yr_window    =min((df[c_name].rolling(extent).min()-df[c_name].rolling(extent).max()).dropna())
    largest_neg_ret_in_1yr_window =min((df[c_name].rolling(extent).min()/df[c_name].rolling(extent).max()-1).dropna())
        
    if(bands):
        if(ln_std):
            ax.fill_between(ts.index,last[0]*np.exp(-ret_std),last[0]*np.exp(ret_std),facecolor='grey',alpha=0.1)
        else:
            ax.fill_between(ts.index,last[0]-ret_std,last[0]+ret_std,facecolor='grey',alpha=0.1)
                
    ax.set_xticklabels(['%s\n%d' % (month_d[x.month], x.year) for x in xtks])
                                            
    #plt.plot(last.index[0].strftime('%m/%d/%y'),last[0],'ro',markersize=10)
    plt.plot(last.index[0],last[0],'ro',markersize=10)
    same_as_last = ts[ts < ts[-1]].tail(1)

    plt.plot(same_as_last.index[0],same_as_last[0],'go',markersize=10)    
    plt.plot(last.index[0],last[0],'ro',markersize=10)

    
    font = FontProperties()
    font.set_family('cursive')
    font.set_style('italic')
    plt.title(c_name + ' [ Retro. Stats -- last:%.2f  max:%.2f  min:%.2f  mean:%.2f  lvl std (1y,all):%.1f,%.1f  %%-ile:%.1f  Z:%.2f]' % (last,ts.max(),ts.min(),ts.mean(),std_1y,lvl_std, pctile,(ts.tail(1)[0]-ts.mean())/ts.std()),size=12,fontproperties=font)

    plt.grid(True,color='.6')

    if(ln_std):
        ax.set_xlabel('Prospective 2sigma (std.lnr) [ -2s: %.2f, +2s: %.2f, return std (ln).: %.1f, min(ts.min(252)/ts.max(252)-1): %.2f]' % (last*np.exp(-2*ret_std),last*np.exp(2*ret_std),100*ret_std,largest_neg_ret_in_1yr_window))
    else:
        ax.set_xlabel('Prospective 2sigma (std.nd) [ -2s: %.2f, +2s: %.2f, return std (norm).: %.1f,min(ts.min(252)-ts.max(252)): %.1f]' % (last-2*ret_std,last+2*ret_std,ret_std,largest_drop_in_1yr_window))
    
    #plt.xlabel(' 3Sig Range [ %s : %s ] --  1sig = %s ' % (ng_wgt,cl_wgt,l.currency(last-3*nrml_ret_std,grouping=True), l.currency(last+3*nrml_ret_std,grouping=True),l.currency(nrml_ret_std,grouping=True)))
    plt.tight_layout(w_pad=1.,h_pad=1.)
        
    #plot_logo(ax,bottomLogo)
    plt.legend(loc='upper right',prop={'size':10})
    plt.show()
    return fig

def px_quick_plot(ts,c_name,ln_std=True,bands=True,freq='AS-JAN',bottomLogo=False):
    month_d={1:'Jan', 2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

    s_dt=ts.index[0]
    e_dt=ts.index[-1]
    extent = min(len(ts),252)
    
    end_dt=e_dt+dt.timedelta(days=90)
    xtks=pd.date_range(start=s_dt,end=end_dt,freq=freq)
    df=pd.DataFrame(ts.values,index=ts.index,columns=[c_name])
    df['mean']= ts.mean()
#    df['max'] = ts.max()
#    df['min'] = ts.min()
    last=df[c_name].tail(1)

    pctile = 100*float(np.sum(ts < ts[-1]))/ts.shape[0]

    lvl_std = np.std(df[c_name])
    std_1y  = np.std(df[c_name].tail(extent))
    df['+-1 zscore %0.1f %0.1f' % (ts.mean()-lvl_std, ts.mean()+lvl_std)] = ts.mean()-lvl_std
    df[''] = ts.mean()+lvl_std
    df['-0.5 zscore %0.1f' % (ts.mean()-0.5*lvl_std)] = ts.mean()-0.5*lvl_std

    fig=px.line(ts,height=15,width=20) #xticks=xtks.to_pydatetime()
    #ymin,ymax = ax.get_ylim()
    #ax.set_ylim([ts.min()-abs(ts.mean()*.1),ts.max()+abs(ts.mean()*.1)])
    #fig=ax.get_figure()
    #fig.set_facecolor('whitesmoke')
    #ax.set_axis_bgcolor('gray')
    
    
#    df_stub=pd.DataFrame(last[0]*np.exp(ret_std),index=pd.date_range(e_dt,end_dt),columns={'+-1 rstd'})
#    df_stub.plot(ax=ax)

    #should check that data is given in daily frequency
    if(ln_std):
        ret_std = np.std(np.log((df[c_name].pct_change()+1)))*np.sqrt(252)
    else:
        ret_std = np.std(df[c_name].diff())*np.sqrt(252)

    largest_drop_in_1yr_window    =min((df[c_name].rolling(extent).min()-df[c_name].rolling(extent).max()).dropna())
    largest_neg_ret_in_1yr_window =min((df[c_name].rolling(extent).min()/df[c_name].rolling(extent).max()-1).dropna())
        
    #if(bands):
    #    if(ln_std):
    #        ax.fill_between(ts.index,last[0]*np.exp(-ret_std),last[0]*np.exp(ret_std),facecolor='grey',alpha=0.1)
    #    else:
    #        ax.fill_between(ts.index,last[0]-ret_std,last[0]+ret_std,facecolor='grey',alpha=0.1)
                
    #ax.set_xticklabels(['%s\n%d' % (month_d[x.month], x.year) for x in xtks])
                                            
    #plt.plot(last.index[0].strftime('%m/%d/%y'),last[0],'ro',markersize=10)
    #plt.plot(last.index[0],last[0],'ro',markersize=10)
    #same_as_last = ts[ts < ts[-1]].tail(1)

    #plt.plot(same_as_last.index[0],same_as_last[0],'go',markersize=10)    
    #plt.plot(last.index[0],last[0],'ro',markersize=10)

    
    font = FontProperties()
    font.set_family('cursive')
    font.set_style('italic')
    #plt.title(c_name + ' [ Retro. Stats -- last:%.2f  max:%.2f  min:%.2f  mean:%.2f  lvl std (1y,all):%.1f,%.1f  %%-ile:%.1f  Z:%.2f]' % (last,ts.max(),ts.min(),ts.mean(),std_1y,lvl_std, pctile,(ts.tail(1)[0]-ts.mean())/ts.std()),size=12,fontproperties=font)

    #plt.grid(True,color='.6')

##    if(ln_std):
##        fig.set_xlabel('Prospective 2sigma (std.lnr) [ -2s: %.2f, +2s: %.2f, return std (ln).: %.1f, min(ts.min(252)/ts.max(252)-1): %.2f]' % (last*np.exp(-2*ret_std),last*np.exp(2*ret_std),100*ret_std,largest_neg_ret_in_1yr_window))
##    else:
##        ax.set_xlabel('Prospective 2sigma (std.nd) [ -2s: %.2f, +2s: %.2f, return std (norm).: %.1f,min(ts.min(252)-ts.max(252)): %.1f]' % (last-2*ret_std,last+2*ret_std,ret_std,largest_drop_in_1yr_window))
    
    #plt.xlabel(' 3Sig Range [ %s : %s ] --  1sig = %s ' % (ng_wgt,cl_wgt,l.currency(last-3*nrml_ret_std,grouping=True), l.currency(last+3*nrml_ret_std,grouping=True),l.currency(nrml_ret_std,grouping=True)))
    #plt.tight_layout(w_pad=1.,h_pad=1.)
        
    #plot_logo(ax,bottomLogo)
    #plt.legend(loc='upper right',prop={'size':10})
    #plt.show()
    return fig

def seasonal_plot(ts,ttle="",normalize=True,average=False,bottomLogo=False):

    df=pd.DataFrame([])
    
    #consider df = pd.DataFrame([])
    for y in np.arange(ts.index.min().year,ts.index.max().year+1):
        #print 'working on %d' % (y)
        ts_y = ts['%d' % (y)]
        
        #create numbered sequenced index, for lookup

        bd_y    = pd.bdate_range(dt.datetime(y,1,1),dt.datetime(y,12,31))
        bd_y_ts = pd.Series(np.arange(0,bd_y.shape[0]),index=bd_y)
        
        #consider what to do for time-series beginning mid-year
        s_bd = bd_y_ts[ts_y.index[0]]
        idx  = s_bd+np.arange(ts_y.index.shape[0])
        
        if normalize:
            v=ts[ts.index.year==y].values/ts[ts.index.year==y].values[0]
        else:
            v=ts[ts.index.year==y].values
            
        df=df.join(pd.DataFrame(v, columns=['%d' % y],index=idx),how='outer')
                
    if average:
        if normalize:
            ax=df[:250].T.mean().plot(label='normalized average')
        else:
            ax=df[:250].T.mean().plot(label='un-normalized average')
    else:
        if normalize:
            ax=df.plot(label='normalized, not-averaged')
        else:
            ax=df.plot(label='un-normalized, not-averaged')
        
    
    plt.title(ttle)

    plt.legend(loc='upper right',prop={'size':10})
    plt.tight_layout(w_pad=0.5,h_pad=0.5)
    #plot_logo(ax,bottomLogo)
    #plt.show()
    return ax.gcf()
    
                
def plot_logo(ax, bottomLogo=False):
    #takes in current axis values and plots rcm logo relative to those values

    with cbook.get_sample_data('C:/Users/tnico/anaconda3/Lib/site-packages/fibonacci-spiral.png') as image_file:
        im = plt.imread(image_file)
    #im = plt.imread('C:\Users\tnico\OneDrive\Logo.png')

    imx1_, imx2_ = ax.get_xlim()
    imy1_, imy2_ = ax.get_ylim()
    
    #print imx1_
    #print imx2_
    #print imy1_
    #print imy2_    
    

    #gets length of x-axis / y-axis, divide by desired scaling factor
    xLength = (imx2_ - imx1_)*0.1
    yLength = (imy2_ - imy1_)*0.1

    #print bottomLogo
    if bottomLogo:
        #for bottom left logo
        ax.imshow(im,extent=[imx1_,imx1_ + xLength,imy1_,imy1_ + yLength],aspect='auto')
    else: 
        #for upper left logo
        ax.imshow(im,extent=[imx1_,imx1_ + xLength,imy2_ - yLength,imy2_],aspect='auto')

    #using "extent" above changes axis values to those of the logo
    #need to reset to original axis values
    ax.set_xlim(imx1_,imx2_)
    ax.set_ylim(imy1_,imy2_)
