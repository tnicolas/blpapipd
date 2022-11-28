import subprocess as s
import psutil
import numpy as np
import blpapi
import datetime
import pandas as pd
import time as t

global rec_cnt
rec_cnt=0


###############################################
# developed by T. Nicolas Steinbach based on
# BBG C++ API and
# Pandas Time Series and DataFrame library
###############################################

from pandas import Series

def initBBGSession(host="LocalHost" , port=8194,start_service=True):
    # Fill SessionOptions
    
    pythons_psutil = []
    for p in psutil.process_iter():
        try:
            if p.name() == 'bbcomm.exe':
                pythons_psutil.append(p)
        except psutil.Error:
            pass
    
    if (np.size(pythons_psutil)==0) and start_service:
        s.Popen('C:\\blp\\API\\bbcomm.exe',shell=True).pid
        #retcode = s.call('C:\\blp\\API\\bbcomm.exe',shell=False)
        
        not_started=True
        while not_started:
            print('Waiting for bbcomm to start')
            t.sleep(2)
            for p in psutil.process_iter():
                try:
                    if p.name() == 'bbcomm.exe':
                        print('Successfully started bbcomm.exe')
                        not_started=False
                except psutil.Error:
                    pass
    else:           
        print("Found bbcomm.exe, no need to launch it")

                        
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(host)
    sessionOptions.setServerPort(port)

    print("Connecting to %s:%s" % (host, port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return
    except:
        session.stop()
#end of openSession
            
    return session
    

def getTimeSeries(session, ticker, fld, sd=[], ed=[], prd="DAILY",cache_override=False):
# check to see if cache file exists for this timeseries
# naming convention:
#    - ticker + fld + sd + ed + prd
#
# store in store_path
   store_path='C:\\Users\\tnico\\OneDrive\\python\\cache\\'

   ticker = ticker.replace('/','')
   if sd == []:
       sd=19500101
   if ed == []:
       ed=int(datetime.datetime.now().strftime('%Y%m%d'))
   
   filename=ticker+' '+ fld+(' %d %d '%(sd,ed))+prd
   try:
       if cache_override:
           raise ""
        
       return_ts = pd.read_pickle('%s%s' % (store_path,filename))
       print("Cache "+filename+" exists.  Loading.")
       return return_ts
    
   except:
       if cache_override:
           print('Retrieving live quote %s, cache_override=True' % (ticker))
       else:
           print(filename+" did not load...creating cache.")
          
       refDataService = session.getService("//blp/refdata")
       #end of openSession

       #build HistoricalDataRequest
       #takes tickers, fields, st_dt end_dt, etc...
       # Create and fill the request for the historical data
       request = refDataService.createRequest("HistoricalDataRequest")

       request.getElement("securities").appendValue(ticker)
       #print "appended " + ticker
       
       request.getElement("fields").appendValue(fld)
       #print "getting   " + fld
        
       request.set("periodicityAdjustment", "ACTUAL")
       request.set("periodicitySelection", prd)
       request.set("startDate",sd)
       request.set("endDate", ed)
       request.set("maxDataPoints", 50000)

       #print "sending request"
       session.sendRequest(request)
       #print "request successful"    
       # process results, pack into a time series object
       
       t_idx = []
       v_srs = []
       
       while(True):
           # We provide timeout to give the chance for Ctrl+C handling:
           ev = session.nextEvent(500)
           for msg in ev:  
               #print 'eventType: %d' % ev.eventType()
               #print 'looking for %d' % blpapi.Event.PARTIAL_RESPONSE   
               #print 'msgType: %s, numElems: %d' % (msg.messageType(), msg.numElements()) 
               
               if ev.eventType() == blpapi.Event.RESPONSE:
                   node=msg.getElement(0)
                   fd=node.getElement('fieldData')
                   #print fd

                   for v in fd.values():
                       t_idx.append(v.getElement('date').getValue())
                       v_srs.append(v.getElement(fld).getValue())
                       
                   result_ts = pd.Series(v_srs,index=pd.to_datetime(t_idx))
                   #,name=ticker+' '+fld)
                   if not cache_override:
                       result_ts.to_pickle('%s%s' % (store_path,filename))
                   
                   return result_ts
               
               elif ev.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                   raise('something went wrong in event retrieve sequence')


def getCLStrikePremiums(session, cls_d, ticker, strike_range):

    # get ATM future (PX_LAST)
    #cl         = getTimeSeries(session, cls_d, cls_d, ticker,'PX_LAST')
    # get ATM i.vol  (HIST_CALL_IMP_VOL)
    #cl_ivol    = getTimeSeries(session, cls_d, cls_d, ticker,'HIST_CALL_IMP_VOL')
    #cl_days_exp= getTimeSeries(session, cls_d, cls_d, ticker + 'C %3.1f Comdty'%atm_k,'FUT_ACT_DAYS_EXP')
    # get N Calls and N Put tickers (ideally, 3 i.vol std dev up and down, in that future's increments)
    # do 10 0.5 pt increments, 10 1 pt increments, then 5 point increments for the rest
    #atm_k= round(cl[0])
    #iv   = cl_ivol[0]/100
    #r0   = arange( round(atm_k-iv*sqrt(cl_days_exp[0]/365.25)/2),round(atm_k+iv*sqrt(cl_days_exp[0]/365.25)/2),0.5 )
    #rL1  = arange( round(atm_k-iv*sqrt(cl_days_exp[0]/365.25)),r0[1],1.0 )
    #rR1  = arange( r0[-1],round(atm_k+iv*sqrt(cl_days_exp[0]/365.25)),1.0 )


# construct request
# structure response into a DataFrame  strike  c/p  pr
# return

    fld="PX_LAST"
    prd="DAILY"
    
    refDataService = session.getService("//blp/refdata")
#end of openSession

#build HistoricalDataRequest
#takes tickers, fields, st_dt end_dt, etc...
        # Create and fill the request for the historical data
    request = refDataService.createRequest("HistoricalDataRequest")

    request.getElement("securities").appendValue("CLZ5C 83.5 Comdty")
    request.getElement("securities").appendValue("CLZ5C 84 Comdty")
    request.getElement("securities").appendValue("CLZ5C 84.5 Comdty")
    request.getElement("securities").appendValue("CLZ5C 85 Comdty")
    request.getElement("securities").appendValue("CLZ5C 85.5 Comdty")
    request.getElement("securities").appendValue("CLZ5C 86 Comdty")
    request.getElement("securities").appendValue("CLZ5C 86.5 Comdty")
    print("appended " + ticker)
            
    request.getElement("fields").appendValue(fld)
    print("getting   " + fld)
        
    request.set("periodicityAdjustment", "ACTUAL")
    request.set("periodicitySelection", prd)
    request.set("startDate",cls_d)
    request.set("endDate", cls_d)
    request.set("maxDataPoints", 1000)
        
    print("sending request")
    session.sendRequest(request)
    print("request successful")
        # process results, pack into a time series object

    t_idx = []
    v_srs = []
    
    while(True):
        # We provide timeout to give the chance for Ctrl+C handling:
        ev = session.nextEvent(500)
        for msg in ev:  
            print('eventType: %d' % ev.eventType())
            print('looking for %d or %d' % (blpapi.Event.PARTIAL_RESPONSE,blpapi.Event.RESPONSE))
                        
            print('msgType: %s, numElems: %d' % (msg.messageType(), msg.numElements()) )
            
            if ev.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                node=msg.getElement(0)
                fd=node.getElement('fieldData')
                print(node)

                for v in fd.values():
                    t_idx.append(v.getElement('date').getValue())
                    v_srs.append(v.getElement(fld).getValue())
                    
                result_ts = Series(v_srs,index=pd.to_datetime(t_idx))
                #return result_ts
            elif ev.eventType() == blpapi.Event.RESPONSE:
                print(node)
                print("Done")
                return result_ts
                
        #elif ev.eventType() == blpapi.Event.PARTIAL_RESPONSE:
        #    throw('something went wrong in event retrieve sequence')

def conditionTS(ts, return_threshold=0.3):
    global rec_cnt
            
    rec_cnt=rec_cnt+1
    print(rec_cnt)
        
    # identify the first major differences
    i_max=ts.diff().dropna().argmax()
    i_min=ts.diff().dropna().argmin()
    
    # is the argmax return greater than threshold?
    if abs(ts.pct_change().dropna()[i_max]) > return_threshold :
        #check to see if argmin is nearby?
        if abs(i_min-i_max) < 4:
            #condition data
            print("conditioning data at %d" % i_max)
            #use average of i_max-1 and i_min+1
            #when indexing into ts, keep in mind that i_max and i_min are for 
            #the shifted, pct_change series
            if i_max > i_min:
                #downspike, min happens before i_max
                ts[i_min+1]=(ts[i_min]+ts[i_max+2])/2
                ts[i_max+1]=ts[i_min+1]
            else:
                #upspike, min happens later than i_max
                ts[i_min+1]=(ts[i_max]+ts[i_min+2])/2
                ts[i_max+1]=ts[i_min+1]
            conditionTS(ts,return_threshold)
        else:
            # i_min and i_max not close enough, return
            return ts
    else:
     return ts
