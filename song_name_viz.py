"""
Este script crea una visualización de la columna 'song_name' en el dataset mostrando las palabras más comunes en los
títulos de las canciones. Para ello, se utiliza la librería WordCloud.
"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from utils import load_data


def main():
    # Cargamos los datos
    X_train, _, _ = load_data()

    # Creamos la nube de palabras
    wordcloud = WordCloud(width=1600, height=800, max_words=100, background_color='white').generate(' '.join(X_train['song_name']))

    # Mostramos la nube de palabras
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
