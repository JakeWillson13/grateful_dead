import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import plotly.express as px

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
# Load and process all Grateful Dead lyrics from CMU archive
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
    df[['word_count', 'avg_word_length', 'unique_word_count']] = pd.DataFrame(metrics.tolist(), index=df.index)
    df['lexical_diversity'] = df['unique_word_count'] / df['word_count'].replace(0, np.nan)
    return df

@st.cache_data
# Load top 50 subset from uploaded CSV file
def load_top50(csv_file):
    top = pd.read_csv(csv_file)
    return top.sort_values('rank').head(50)

@st.cache_data
# Fetch top 50 from GitHub raw CSV
def load_top50_from_url():
    top = pd.read_csv(CSV_URL)
    return top.sort_values('rank').head(50)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Grateful Dead Lyrics Dashboard", layout="wide")
    st.title("Grateful Dead Lyrics Analysis")
    st.markdown("Exploring lyric complexity vs popularity across Grateful Dead songs.")

    # Scrape and process lyrics
    with st.spinner("Scraping and processing lyrics..."):
        df_lyrics = load_lyrics()

    # Display all songs metrics table
    st.header("Word Analysis Metrics")
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
    fig_all.update_layout(title='Unique vs Total Words (All Songs)', margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig_all, use_container_width=True)

    # Load Top 50 data
    with st.spinner("Loading Top 50 songs..."):
        if csv_file:
            top50 = load_top50(csv_file)
        else:
            top50 = load_top50_from_url()
        merged = pd.merge(
            top50,
            df_lyrics,
            how='inner',
            left_on='Title',
            right_on='title'
        )

    # Scatter for top 50 songs
    st.subheader("Top 50 Songs by Spotify Streams")
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
    fig_top.update_layout(title='Top 50 Grateful Dead Songs: Unique vs Total Words', margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig_top, use_container_width=True)

if __name__ == '__main__':
    main()
