
from flask import Flask,render_template,request,redirect,url_for
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from pulp import *

app=Flask(__name__) 


#Gets the lists for plotting the histogram
def for_hist(var,num,col):
    var_bins = pd.cut(var, 
                        bins=np.arange(0, int(max(var))+num+1,num), right=False,
                        labels=[i for i in range(0,int(max(var))+1,num)])

    count = pd.Series.groupby(var_bins,by=var_bins,).count().reset_index(name='freq')
    x = list(count[col])
    freq = list(count['freq'])
    return x,freq

#Gets coordinates for the scatter plot
def bivar_list(var1, var2):
    newlist=[]
    for x, y in zip(var1, var2):
        newlist.append({'x': x, 'y': y})
    newlist = str(newlist).replace('\'', '')
    return newlist

#Calculates lags for the petrol prices
def prices_df(p_df1):
    p_df1["Date"] = p_df1[p_df1.columns[0:3]].apply(     
    lambda x: ' '.join(x.astype(str)),
    axis=1)
    p_df1 = pd.concat([p_df1['Date'],p_df1['Delhi']],axis=1)
    p_df1.set_index('Date',inplace= True)    
    p_df1['lag3']=p_df1.Delhi.shift(3)
    p_df1['lag5']=p_df1.Delhi.shift(5)
    p_df1['lag7']=p_df1.Delhi.shift(7)
    return p_df1

#Gives the lists of dates, actual prices and lag values; n-lag
def get_lags(p_df2, n):
    x = list(p_df2.index[int(n):])
    y1 = list(p_df2.Delhi[int(n):])
    y2= list(p_df2['lag'+str(n)].iloc[int(n):])
    return x,y1,y2

#Predicts today's petrol price
def lin_reg(x,y, ch="nil"):   
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    n = int(0.8*len(x))
    X_train = x[:n]
    y_train = y[:n] 
    if ch=="opt":
        test = x[-6].reshape((-1, 1))
    else:
        test=x
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(test)
    y_pred = y_pred.reshape(1,-1)
    return y_pred[0]

#Calculates optimal refill quantity and period
def optimal(user):
    ans=[]
    err= "nil"
    p=0
    p_df = prices_df(prices)
    day,y_act,y_lag = get_lags(p_df,n=5)
    petrol=lin_reg(x=y_lag, y=y_act, ch="opt")[0]

    prob=LpProblem("Fuel",LpMaximize)
    x1=LpVariable("x1",lowBound=1, cat='Integer')
    Q = x1*user["D_dist"]/user["Mileage"]
    prob += Q
    prob += petrol*Q <= user['Amt']
    prob += Q <= user["Tank_cap"]
    prob += -Q <= -user["Reserve_cap"]
    status=prob.solve()
    if LpStatus[status]=='Optimal':
        n = int(value(x1))
        Q1 = round(value(prob.objective),2)
        p = round(petrol*Q1)
        ans = [n,Q1]
        return ans, err, p 
    else:
        err = "Inconsistent data"
        return ans, err, p


#User data
#user_df = pd.read_csv("C:\Users\raksh\OneDrive\Desktop\Final\FINAL FINAL\userdata1.csv")

#Petrol prices
prices = pd.read_csv("C:\\Users\\raksh\\OneDrive\\Desktop\\Final\\FINAL FINAL\\petrolprices.csv")


#HOMEPAGE
@app.route("/",methods=["GET","POST"])
def home():
    return render_template("home.html")

#REGISTRATION PAGE
@app.route("/reg",methods=["GET","POST"])
def reg():
    return render_template("reg.html")


#MAIN OUTPUT PAGE
@app.route("/result",methods=["GET","POST"]) #new
def result():
    user_df = pd.read_csv("C:\\Users\\raksh\\OneDrive\\Desktop\\Final\\FINAL FINAL\\userdata1.csv")
    if request.method == "POST":
        user = {
            'Sl.no': len(user_df)+1,
            # 'Username': request.form["username"],
            'Vehicle': request.form["vm"],
            'Mileage': float(request.form["mil"]),
            'Tank_cap': float(request.form["tc"]),
            'Reserve_cap': float(request.form["rc"]),
            'Engine_cap': float(request.form["ec"]),
            'D_dist': float(request.form["dist"]),
            'W_dist': float(request.form["m_dist"]),
            'Refill_period': int(request.form["period"]),
            'Last_refill': int(request.form["lst_day"]),
            'Refill_qty': float(request.form["rqty"]),
            'Amt': float(request.form["amt"])
            }
    
    result,msg,p = optimal(user)
    if(msg=="nil"):
        user_df=user_df.append(user, ignore_index=True)
        user_df.to_csv('C:\\Users\\raksh\\OneDrive\\Desktop\\Final\\FINAL FINAL\\userdata1.csv', index = False)

    return render_template("result.htm", res=result, msg=msg, price=p)

#EDA - UNIVARIATE
@app.route("/unieda", methods = ["GET", "POST"])
def univariate():
    user_df = pd.read_csv("C:\\Users\\raksh\\OneDrive\\Desktop\\Final\\FINAL FINAL\\userdata1.csv")
    Mil,Mil_freq = for_hist(user_df['Mileage'], num=5, col="Mileage")

    tank, tank_freq = for_hist(user_df['Tank_cap'], num=5, col="Tank_cap")

    dist, dist_freq = for_hist(user_df['D_dist'], num=10, col="D_dist")

    Amtt, Amt_freq = for_hist(user_df['Amt'], num=100, col="Amt")

    Refill_freq_count = user_df.groupby('Refill_period')['Refill_period'].count().reset_index(name='freq')
    refill_period = list(Refill_freq_count['Refill_period'])
    ref_freq = list(Refill_freq_count['freq'])   

    return render_template('univariate.html', Mil1=Mil, Mil_freq=Mil_freq, tank=tank, tank_freq=tank_freq, dist=dist, dist_freq=dist_freq, 
                                            Amt=Amtt, Amt_freq=Amt_freq, refill_period=refill_period, ref_freq=ref_freq)


#EDA - BIVARIATE
@app.route("/bieda", methods = ["GET", "POST"])
def bivariate():
    user_df = pd.read_csv("C:\\Users\\raksh\\OneDrive\\Desktop\\Final\\FINAL FINAL\\userdata1.csv")
    T_cap = list(user_df['Tank_cap'])
    E_cap = list(user_df['Engine_cap'])
    Mil = list(user_df['Mileage'])  
    Amt_bud = list(user_df['Amt'])
    Refill = list(user_df['Refill_period'])
    
    T_M = bivar_list(T_cap,Mil)
    E_M = bivar_list(E_cap,Mil)
    Amt_M = bivar_list(Mil, Amt_bud)
    Refill_M = bivar_list(Mil, Refill)

    return render_template('bivariate.html', Tcap_M=T_M, Ecap_M=E_M, Amt_M=Amt_M, Refill_M=Refill_M)


#EDA - LAGS
@app.route("/lags", methods = ["GET", "POST"])
def lags():
    p_prices = prices_df(prices)
    x3, y_act3, y_lag3 = get_lags(p_prices,3)
    y_pred3 = list(lin_reg(x=y_lag3, y=y_act3))
    x5, y_act5, y_lag5 = get_lags(p_prices,5)
    y_pred5 = list(lin_reg(x=y_lag5, y=y_act5))
    x7, y_act7, y_lag7 = get_lags(p_prices,7)
    y_pred7 = list(lin_reg(x=y_lag7, y=y_act7))

    return render_template('lagss.html', x3=x3, y_act3=y_act3, y_pred3=y_pred3, x5=x5, y_act5=y_act5, y_pred5=y_pred5, x7=x7, y_act7=y_act7, y_pred7=y_pred7)

# @app.route("/lags",methods=["GET","POST"]) #new
# def lags():
#     return render_template("lags.html")


#ABOUT
@app.route("/about", methods = ["GET", "POST"])
def about():
    return render_template('about.html')

@app.route("/hmm", methods = ["GET", "POST"])
def blah():
    nam = {'pls': user_df.columns.to_list()}
    return nam


if __name__=="__main__":
    app.run(debug=True)
