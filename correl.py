import yfinance as yf
import pandas as pd
import streamlit as st
import pandas_ta as ta
import statsmodels.api as sm
from datetime import datetime
import os

st.title("Broadcom & VMware Stock Correlation Analysis")
st.write('Broadcom announced their intent to acquire VMware on May 2nd, 2022. As expected, there has been a lot of speculation \
         regarding the closure of this acquisition. By analyzing these charts, you can gain additional insights into the \
         market\'s sentiment and expectations at any point during the 1+ year since the announcement. You can also examine \
         how news or changes in sentiment may have affected the correlation between these two stocks.\n')
start_date = '2022-03-02' # 2 months before the announcement
today = datetime.today().strftime("%Y-%m-%d")
end_date = today

# If we have saved data files, we use those first.
if os.path.exists('avgo.pkl') & os.path.exists('vmw.pkl') & os.path.exists('nasdaq.pkl'):
    modification_date = datetime.fromtimestamp(os.path.getmtime('avgo.pkl')).strftime("%Y-%m-%d") # we only use 1 file for simplicity
    files_exist = True
else:
    modification_date = start_date # totally arbitrary
    files_exist = False

if ((files_exist & (modification_date != today)) | (not files_exist)):
    avgo = yf.download('AVGO', start=start_date, end=end_date)
    vmw = yf.download('VMW', start=start_date, end=end_date)
    nasdaq = yf.download('^IXIC', start=start_date, end=end_date) # Symbol for NASDAQ

    # Save them for future reference
    avgo.to_pickle("avgo.pkl")
    vmw.to_pickle("vmw.pkl")
    nasdaq.to_pickle("nasdaq.pkl")
else:
    avgo = pd.read_pickle("avgo.pkl")
    vmw = pd.read_pickle("vmw.pkl")
    nasdaq = pd.read_pickle("nasdaq.pkl")


avgo['VWAP'] = ta.vwap(avgo['High'], avgo['Low'], avgo['Close'], avgo['Volume'])
vmw['VWAP'] = ta.vwap(vmw['High'], vmw['Low'], vmw['Close'], vmw['Volume'])

avgo['Return'] = avgo['Close'].pct_change()
vmw['Return'] = vmw['Close'].pct_change()
nasdaq['Return'] = nasdaq['Close'].pct_change()

price_corr = avgo['Close'].rolling(20).corr(vmw['Close'])
vwap_corr = avgo['VWAP'].rolling(20).corr(vmw['VWAP'])
return_corr = avgo['Return'].rolling(20).corr(vmw['Return']).rename('Return correlation')

st.subheader("Price Action & VWAP Correlation")
st.write('This chart shows two curves, the regular price correlation and the correlation of their respective\
         VWAPs. The VWAP, or volume-weighted average price, is moving average calculation that incorporates\
         the volume in its fomula, giving you a more holistic view of the daily movements. We use the 20-day\
         rolling VWAP calculation here.\nYou can observe the negligible impact of volume in the correlation.')

correlations = pd.DataFrame({
    'Price Action Correlation': price_corr,
    'VWAP Correlation': vwap_corr
})
st.line_chart(correlations)

st.subheader('Correlation of Daily Returns')
st.write('Now we chart the correlation between the daily returns of both stocks. Daily returns are the difference between the\
         price at close for the day, and the price at close for the day before.')
st.line_chart(return_corr)

# Detrended correlations calculations
X = nasdaq['Return'][1:].values.reshape(-1,1)  # independent variable
y_avgo = avgo['Return'][1:]  # dependent variable
y_vmw = vmw['Return'][1:]  # dependent variable
avgo_vwap_return = avgo['VWAP'].pct_change()[1:]
vmw_vwap_return = vmw['VWAP'].pct_change()[1:]

#OLS Regressions
ols_avgo = sm.OLS(y_avgo, sm.add_constant(X)).fit()
avgo_resid = ols_avgo.resid
ols_vmw = sm.OLS(y_vmw, sm.add_constant(X)).fit()
vmw_resid = ols_vmw.resid
detrended_price_corr = avgo_resid.rolling(20).corr(vmw_resid).rename('Det. Price Correlation')

ols_avgo_vwap = sm.OLS(avgo_vwap_return, sm.add_constant(X)).fit()
ols_vmw_vwap = sm.OLS(vmw_vwap_return, sm.add_constant(X)).fit()

avgo_vwap_resid = ols_avgo_vwap.resid
vmw_vwap_resid = ols_vmw_vwap.resid

st.subheader("Detrended Price Action Correlation")
st.write('Both, Broadcom and VMware are traded in the NASDAQ. To remove or minimize the influence of the overall market index in the correlation \
         calculation, we utilize a technique called _detrending_. Detrending allows you to eliminate \
         the common trends and variations that are shared by both stocks and the market index. One common approach is to \
         calculate the correlation between the residuals obtained from a linear regression of the stocks\' returns on the \
         market index\'s returns. By regressing the individual stock returns against the market index returns, we can extract \
         the unique, idiosyncratic components of the stock returns that are not related to the overall market movements.\
         ')

st.line_chart(detrended_price_corr)

detrended_vwap_corr = avgo_vwap_resid.rolling(20).corr(vmw_vwap_resid).rename('Det. VWAP Correlation')
st.subheader("Detrended VWAP Correlation")
st.line_chart(detrended_vwap_corr)

st.subheader("Detrended Price Action and VWAP Correlations Together")
detrended_correlations = pd.DataFrame({
    'Det. Price Correlation': detrended_price_corr,
    'Det. VWAP Correlation': detrended_vwap_corr
})
st.line_chart(detrended_correlations)

st.subheader('Conclusions')
st.write('Whether this acquisition will ultimately be completed or not is not the focus of this analysis, as the reality \
         is much more complex than what these charts can reveal. Divergences between the stocks can occur at any time due \
         to a variety of factors. It is crucial to understand that this information is not intended as investment advice \
         and is provided \'as is\'. I cannot guarantee its accuracy or provide any representations or assurances.\n')

st.write('\n\nProgram written by [Ramiro Salas](https://ramirosalas.com). [Source code available in GitHub](https://github.com/RamXX)')