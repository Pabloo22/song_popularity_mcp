"""
Hemos decidido incluir únicamente aquellas palabras o conjuntos de ellas que se encuentren
en al menos un 5% de las palabras. Por tanto, incluimos las siguientes variables
binarias: contains_hyphen, contains_feat, is_derived (si es una canción creada a partir de otra)
y contains_parentheses. Además de esto, añadiremos columnas con la siguiente información:
número de sílabas (num_syllables), número medio de caracteres por palabra,
número de palabras que empiezan por mayúscula (num_uppercase_words) y
por minúscula (num_lowercase_words). Además, realizaremos un análisis de sentimiento y
una detección del idioma.
"""
import langdetect
import numpy as np
import pyphen
import textblob

from utils import load_data


def detect_lang(text):
    try:
        return langdetect.detect(text)
    except langdetect.LangDetectException:
        print(f'{text} could not be detected')
        return 'unknown'


SUPPORTED_LANGS = {'af_za': 'af_ZA', 'af': 'af', 'be_by': 'be_BY', 'be': 'be', 'bg_bg': 'bg_BG', 'bg': 'bg',
                   'ca': 'ca', 'cs_cz': 'cs_CZ', 'cs': 'cs', 'da_dk': 'da_DK', 'da': 'da', 'de_at': 'de_AT',
                   'de': 'de', 'de_ch': 'de_CH', 'de_de': 'de_DE', 'el_gr': 'el_GR', 'el': 'el', 'en_gb': 'en_GB',
                   'en': 'en', 'en_us': 'en_US', 'eo': 'eo', 'es': 'es', 'et_ee': 'et_EE', 'et': 'et', 'fr': 'fr',
                   'gl': 'gl', 'hr_hr': 'hr_HR', 'hr': 'hr', 'hu_hu': 'hu_HU', 'hu': 'hu', 'id_id': 'id_ID',
                   'id': 'id', 'is': 'is', 'it_it': 'it_IT', 'it': 'it', 'lt': 'lt', 'lv_lv': 'lv_LV', 'lv': 'lv',
                   'mn_mn': 'mn_MN', 'mn': 'mn', 'nb_no': 'nb_NO', 'nb': 'nb', 'nl_nl': 'nl_NL', 'nl': 'nl',
                   'nn_no': 'nn_NO', 'nn': 'nn', 'pl_pl': 'pl_PL', 'pl': 'pl', 'pt_br': 'pt_BR', 'pt': 'pt',
                   'pt_pt': 'pt_PT', 'ro_ro': 'ro_RO', 'ro': 'ro', 'ru_ru': 'ru_RU', 'ru': 'ru', 'sk_sk': 'sk_SK',
                   'sk': 'sk', 'sl_si': 'sl_SI', 'sl': 'sl', 'sq_al': 'sq_AL', 'sq': 'sq', 'sr': 'sr',
                   'sr_latn': 'sr_Latn', 'sv': 'sv', 'te_in': 'te_IN', 'te': 'te', 'uk_ua': 'uk_UA', 'uk': 'uk',
                   'zu_za': 'zu_ZA', 'zu': 'zu'}


def count_syllables(row):
    words = row['song_name_without_parentheses']
    lang = row['language']
    if lang not in SUPPORTED_LANGS.keys():
        return np.nan
    dic = pyphen.Pyphen(lang=lang)
    num_syllables = 0
    for word in words.split():
        num_syllables += len(dic.inserted(word).split('-'))
    return num_syllables


def create_new_features(df):
    song_names = df['song_name']
    df['contains_hyphen'] = song_names.str.contains('-').astype(int)
    df['contains_feat'] = song_names.str.lower().str.contains('feat').astype(int)
    df['is_derived'] = song_names.str.lower().str.contains('version|remix|remastered').astype(int)
    df['contains_parentheses'] = song_names.str.contains('\(|\)').astype(int)

    df['language'] = song_names.swifter.apply(detect_lang)

    df_aux = df[['song_name', 'language']].copy()
    # Eliminamos el contenido que está dentro de paréntesis, corchetes o '-', puesto que no forma parte del título
    df_aux['song_name_without_parentheses'] = df_aux['song_name'].str.split('\(|\)|\[|\]|-').str[0]
    df['num_syllables'] = df_aux.swifter.apply(count_syllables, axis=1)

    num_words = df_aux['song_name_without_parentheses'].str.split().str.len()
    num_chars = df_aux['song_name_without_parentheses'].str.len()
    df['avg_chars_per_word'] = num_chars / num_words
    df['num_uppercase_words'] = df['song_name'].str.count(r'[A-Z][a-z]+')
    df['num_lowercase_words'] = num_words - df['num_uppercase_words']

    df['sentiment'] = df['song_name'].swifter.apply(lambda x: textblob.TextBlob(x).sentiment.polarity)

    return df


def main():
    train, X_test = load_data(split=False, no_duplicates=True)

    train = create_new_features(train)
    X_test = create_new_features(X_test)

    train.to_csv('data/train_v2.csv', index=False)
    X_test.to_csv('data/test_v2.csv', index=False)


if __name__ == '__main__':
    main()
