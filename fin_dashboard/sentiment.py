import pandas as pd
import nltk
import plotly.express as px
from gnews import GNews

nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_analysis(stock):

    # Get news data from GNews
    news = GNews().get_news(stock)
    data = pd.DataFrame(news)
    data['published date'] = pd.to_datetime(data['published date'])

    data = data.drop(['description', 'url', 'publisher'], axis=1)
    data = data.rename(columns={'published date':'date'})

    data['date'] = data['date'].dt.tz_localize(None)

    # Initialize VADER 
    vader = SentimentIntensityAnalyzer()
    
    scores = data['title'].apply(vader.polarity_scores).tolist()

    scores_df = pd.DataFrame(scores)

    clean_data = data.join(scores_df, rsuffix='_right')             
    clean_data = clean_data.set_index('date')    
    clean_data = clean_data.rename(columns={"compound": "sentiment_score"})

    clean_data = clean_data[clean_data['sentiment_score'] != 0]

    score = clean_data.drop(['neg', 'neu', 'pos'], axis=1)

    def pos_neg(row):
        if row['sentiment_score'] > 0:
            return 'Positive'
        elif row['sentiment_score'] < 0:
            return 'Negative'
        
    score['pos_neg'] = score.apply(lambda row: pos_neg(row),  axis=1)

    mean_scores = clean_data.resample('D').mean()

    # Plot the sentiment graph
    fig = px.bar(mean_scores, x = mean_scores.index, y = 'sentiment_score', title = stock + ' Sentiment Scores')

    return fig, score