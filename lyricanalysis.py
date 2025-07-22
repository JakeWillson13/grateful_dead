import matplotlib.cm as cm
import numpy as np

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


