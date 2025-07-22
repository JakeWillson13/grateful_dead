import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# URL to raw CSV on GitHub
CSV_URL = "https://raw.githubusercontent.com/JakeWillson13/grateful_dead/main/gratefuldead.csv"

# Function to compute text metrics
def text_metrics(lyrics: str):
    words = re.findall(r"\b\w+\b", lyrics.lower())
    n = len(words)
    if n == 0:
        return 0, 0.0, 0
    lengths = [len(w) for w in words]
    return n, np.mean(lengths), len(set(words))

@st.cache_data
def load_lyrics():
    BASE = 'https://www.cs.cmu.edu/~mleone/'
    idx = requests.get(BASE + 'dead-lyrics.html').text
    links = BeautifulSoup(idx, 'html.parser').find_all('a', href=True)
    urls = [BASE + a['href'] for a in links if a['href'].endswith('.txt')]

    rows = []
    for u in urls:
        title = u.rsplit('/', 1)[-1][:-4].replace('_', ' ')
        lyrics = requests.get(u).text
        rows.append({'title': title, 'lyrics': lyrics})
    df = pd.DataFrame(rows)

    metrics = df['lyrics'].map(text_metrics)
    df[['word_count', 'avg_word_length', 'unique_word_count']] = pd.DataFrame(
        metrics.tolist(), index=df.index
    )
    df['lexical_diversity'] = df['unique_word_count'] / df['word_count'].replace(0, np.nan)
    return df

@st.cache_data
def load_top50_from_url():
    top = pd.read_csv(CSV_URL)
    return top.sort_values('rank').head(50)

def main():
    st.set_page_config(page_title="Grateful Dead Lyrics Dashboard", layout="wide")
    st.title("Grateful Dead Lyric Analysis")
    st.markdown("Explore lyric complexity vs popularity across Grateful Dead songs.")

    # Scrape and process lyrics
    with st.spinner("Scraping and processing lyrics..."):
        df_lyrics = load_lyrics()

    # Display all songs metrics table
    st.header("All Songs Metrics")
    st.dataframe(
        df_lyrics[['title', 'word_count', 'avg_word_length', 'unique_word_count', 'lexical_diversity']]
            .sort_values('word_count', ascending=False)
            .rename(columns={
                'title': 'Title',
                'word_count': 'Total Word Count',
                'avg_word_length': 'Avg Word Length',
                'unique_word_count': 'Unique Word Count',
                'lexical_diversity': 'Lexical Diversity'
            }),
        use_container_width=True
    )

    # Scatter for all songs
    st.subheader("Lyrical Analysis Metrics (309 Songs)")
    fig_all = px.scatter(
        df_lyrics,
        x='word_count', y='unique_word_count',
        hover_name='title', color='lexical_diversity',
        size='avg_word_length',
        labels={
            'word_count': 'Total Word Count',
            'unique_word_count': 'Unique Word Count',
            'lexical_diversity': 'Lexical Diversity',
            'avg_word_length': 'Avg Word Length'
        },
        size_max=20
    )
    fig_all.update_layout(
        title='Unique Words vs Total Word Count',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # Load and merge Top 50 data
    with st.spinner("Loading Top 50 songs..."):
        top50 = load_top50_from_url()
        merged = pd.merge(
            top50,
            df_lyrics,
            how='inner',
            left_on='Title',
            right_on='title'
        )

    # Scatter for top 50 songs
    st.subheader("Unique vs Total Word Count (Top 50 Songs)")
    fig_top = px.scatter(
        merged,
        x='word_count', y='unique_word_count',
        size='rank', color='lexical_diversity',
        color_continuous_scale='Plasma',
        hover_name='Title',
        labels={
            'word_count': 'Total Word Count',
            'unique_word_count': 'Unique Word Count',
            'rank': 'Popularity Score',
            'lexical_diversity': 'Lexical Diversity'
        },
        size_max=25
    )
    fig_top.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='white')))
    fig_top.update_layout(
        title='Top 50 Grateful Dead Songs: Unique vs Total Words',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_top, use_container_width=True)

# Word Cloud for Top 50 Lyrics
st.subheader("Word Cloud: Top 50 Songs")

# Combine all lyrics
all_lyrics_top50 = ' '.join(merged['lyrics'].tolist())

# Tokenize and filter
words_top50 = re.findall(r'\b\w+\b', all_lyrics_top50.lower())
stopwords_set = set(STOPWORDS)
custom_stopwords = {
    'like','know','don','t','got','get','gotta','come','going','gonna',
    'said','just','one','see','well','little','say','man','can','back',
    'tell','never','always','around','dead','grateful'
}
stopwords_set.update(custom_stopwords)

filtered_words = [w for w in words_top50 if w not in stopwords_set]
word_counts = Counter(filtered_words)

# Create color function using Plasma colormap
plasma = cm.get_cmap('plasma')

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    normalized = font_size / max(word_counts.values())
    rgba = plasma(normalized)
    return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.8)"

# Generate word cloud
wc = WordCloud(
    width=800,
    height=400,
    background_color=None,
    mode='RGBA',  # enables transparency
    stopwords=stopwords_set,
    min_font_size=10,
    color_func=color_func
).generate_from_frequencies(word_counts)

# Plot and display
fig_wc, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig_wc)

if __name__ == '__main__':
    main()


