import requests


def analyst_recommendation(ticker):

    lhs_url = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
    rhs_url = (
        "?formatted=true&crumb=swg7qs5y9UP&lang=en-US&region=US&"
        "modules=upgradeDowngradeHistory,recommendationTrend,"
        "financialData,earningsHistory,earningsTrend,industryTrend&"
        "corsDomain=finance.yahoo.com"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }
    url = lhs_url + ticker + rhs_url
    try:

        r = requests.get(url, headers=headers)
        result = r.json()["quoteSummary"]["result"][0]

        reco_1 = result["financialData"]["targetMeanPrice"]["fmt"].replace(",", "")
        # reco_2 = result["financialData"]["targetHighPrice"]["fmt"].replace(",", "")

        reco_1 = float(reco_1)
        # reco_2 = float(reco_2)

        return (reco_1)

    except:
        return ("This ticker does not have Analyst Recommendations")
