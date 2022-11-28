import pandas as pd
import numpy as np
import ffn as f
import matplotlib.pyplot as plt
from io import StringIO
import prettytable

import blpapipd.blpapipd as bdu

from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.font_manager import FontProperties


def make_prettytable(s,format=True):
    # make this a function...reuse w/ calls to describe
    output = StringIO()
    if format:
        s = s.as_format('.3f')
    s.to_csv(output)
    output.seek(0)
    pt = prettytable.from_csv(output)
    pt.align='r'
    return pt


def plot_voltron(vi,pdf_pages):
    u_name     = vi.u
    u_df       = vi.u_df
    ivol_name  = vi.ivol[:12] #limit the i.vol name to no more than 12 chars
    ivol_df    = vi.ivolp_df[['orig','clnd','rgrsd']]
    OLS_r      = vi.OLS_r
    ADF_r      = vi.ADF_r

    # take a voltron result dictionary
    # {'under': under_df (w/ columns 'orig','clnd')
    # plot underlying w/ fixes, generate return stats
    
    # plot i.vol w/ fixes & extended regression extension
    
    # assemble description stats, OLS report, ADF test results, calc_stas results
    # also post any time periods that were too large and caused algo to drop it
    
    # possibly work on a data integrity inventory
    
    # show plot or print to PDF
    
    fig=plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(1, 2, 2)
    ax2 = fig.add_subplot(2, 2, 1)
    ax3 = fig.add_subplot(2, 2, 3)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    
    u_df.plot(ax=ax2,color=('forestgreen','coral','cornflowerblue'))
    font = FontProperties()
    font.set_family('cursive')
    font.set_style('italic')
    
    ax2.set_title('%s original & cleaned data' % (u_name),fontproperties=font)
    ivol_df.plot(ax=ax3,color=('forestgreen','coral','cornflowerblue'))
    ax3.set_title('%s original, cleaned & regressed backfill' % (ivol_name),fontproperties=font)
    
    r=u_df['clnd'].calc_stats()
    s=r.stats
    
    s[0:2]=s[0:2].apply(lambda x: x.strftime('%m-%d-%Y'))
    s[2:] =r.stats[2:].as_format('.3f')
    
    s.index=['%021s' % k for k in s.index]
    S=pd.Series(data=['Value'],index=['%s Stats' % u_name])
    s=S.append(s)
    
    pt1 = make_prettytable(s,format=False)
    
    r=ivol_df['rgrsd'].dropna().calc_stats()
    s=r.stats
    
    s[0:2]=s[0:2].apply(lambda x: x.strftime('%m-%d-%Y'))
    s[2:] =r.stats[2:].as_format('.3f')
    
    s.index=['%021s' % k for k in s.index]
    S=pd.Series(data=['Value'],index=['%s Stats' % ivol_name])
    s=S.append(s)
    
    pt4 = make_prettytable(s,format=False)

    
    #eventually format the tables better
    pt2 = make_prettytable(pd.DataFrame(u_df['clnd'].values,index=u_df['clnd'].index,columns=['%s' % u_name]).describe())
    pt3 = make_prettytable(pd.DataFrame(ivol_df['rgrsd'].values,index=ivol_df['rgrsd'].index,columns=['%s' % ivol_name]).describe())


    #start setting page from the bottom

    # set calc_stats (return) report on page
    ax1.text(0.01,0.005,pt1,fontsize=9,family='monospace')
    ax1.text(0.51,0.005,pt4,fontsize=9,family='monospace')
    
    # set OLS & ADF reports here
    ax1.text(0.07,0.58,OLS_r.summary(),fontsize=8,family='monospace')
    ax1.text(0.09,0.56,'ADF Test of Resids.:'+ '%.3f {5%% %.4f, 1%% %.4f' % (ADF_r[0], ADF_r[4]['5%'], ADF_r[4]['1%']),fontsize=8,family='monospace')
    
    # set the dataframe stats via describe, enhance this to include 3-sigma, z-score, pctile?
    ax1.text(0.075,0.86,pt2,fontsize=10,family='monospace')
    ax1.text(0.525,0.86,pt3,fontsize=10,family='monospace')
    
    plt.tight_layout()
    plt.show()
    pdf_pages.savefig()
        

# def main():
#     s=bdu.initBBGSession()
#     spx=bdu.getTimeSeries(s,'SPX Index','PX_LAST')
#     spx_ivol=bdu.getTimeSeries(s,'SPX Index','12MTH_IMPVOL_100.0%MNY_DF')

#     spx2=spx
#     spx2.name='clnd'
    
#     #test drive plot_voltron
#     d = {'u_name':'SPX Index',
#          'u_df':pd.concat([spx,spx2]),
#          'ivol_name':'12 mo. i.vol',
#          'ivol_df':pd.DataFrame(spx_ivol,columns=['rgrsd'])}
#     print(d)
#     plot_voltron(d, PdfPages)
                    
# if __name__ == "__main__":
    
#     main()