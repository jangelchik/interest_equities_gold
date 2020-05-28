## Price of gold versus top central bank overnight deposit rate and top global stock indices by market capitalization:

#### Domain research sources:
Top forex pairs (to determine top global currencies by trading volume): https://www.ig.com/us/trading-strategies/top-10-most-traded-currency-pairs-191206
Use this information to determine which Central Bank rates to consider
Top stock market indices by market capitalization, per Statista: https://www.statista.com/statistics/270126/largest-stock-exchange-operators-by-market-capitalization-of-listed-companies/

#### Features
Daily Central bank rates on interbank overnight loans: https://www.global-rates.com/interest-rates/central-banks/central-banks.aspx
- The Fed (USD): https://fred.stlouisfed.org/series/FEDFUNDS
- ECB (EUR): https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html
- BoJ (JPY): https://fred.stlouisfed.org/series/IRSTCI01JPM156N
- United Kingdom (GBP): https://fred.stlouisfed.org/series/GBPONTD156N
- Australia (AUD): https://www.rba.gov.au/statistics/cash-rate/ - Need to scrape

Daily stock index performance
- NYSE Composite Index history: https://finance.yahoo.com/quote/%5ENYA?p=^NYA&.tsrc=fin-srch
- NASDAQ Composite Index history: https://finance.yahoo.com/quote/%5EIXIC?p=^IXIC&.tsrc=fin-srch
- Japan Exchange Group Composite Index history: https://finance.yahoo.com/quote/8697.T?p=8697.T&.tsrc=fin-srch
- Shanghai Stock Exchange Composite Index history: https://finance.yahoo.com/quote/%5ESSEC?p=^SSEC&.tsrc=fin-srch
- Hang Seng (Hong Kong) Index History: https://finance.yahoo.com/quote/%5EHSI?p=^HSI&.tsrc=fin-srch

#### Target
Price of gold: https://www.usagold.com/reference/prices/goldhistory.php?ddYears=2018 - Need to scrape

#### Model
Because all features are continuous variables, and I am interested in predicting a continuous value (price of gold), I will be using a Gradient Boosted Regressor.



```python

```
