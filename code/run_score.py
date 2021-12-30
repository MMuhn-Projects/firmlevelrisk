# Modules
import sys

sys.path.append("./code/")
import _helpers as h
import re
import pandas as pd
import os

# Files
input_dir = "input/cases/"
output_dir = "output/"
output_fname = input_dir.split("/")[-2]
sentimentwords_file = (
    "input/sentimentwords/" + "LoughranMcDonald_MasterDictionary_2018.csv"
)
riskwords_file = "input/riskwords/synonyms.txt"
polbigrams_file = "input/political_bigrams/political_bigrams.csv"


"""1) Load auxiliary data sets"""

# Import positive and negative sentiment words, risk words, and collect all
sentiment_words = h.import_sentimentwords(sentimentwords_file)
risk_words = h.import_riskwords(riskwords_file)
allwords = dict(sentiment_words, **{"risk": risk_words})

# Import political bigrams
political_bigrams = h.import_politicalbigrams(polbigrams_file)

# SarsCov2-related words
sarscov2 = [
    "Coronavirus",
    "Corona virus",
    "coronavirus",
    "Covid-19",
    "COVID-19",
    "Covid19",
    "COVID19",
    "SARS-CoV-2",
    "2019-nCoV",
]
sarscov2_words = set([re.sub("[^a-z ]", "", x.lower()) for x in sarscov2])

"""2) List case files"""

# List cases in the input_dir
input_files = os.listdir(input_dir)[0:100]
# Parse them to return the id and opinion_num
all_files = [h.parse_file_name(file) for file in input_files]

"""3) Scoring"""

# Loop through files
scores_list = []
for file in all_files:

    # Read text
    file_path = input_dir + file["fname"]
    with open(file_path, "r") as f:
        case_text = f.read()

    # Preprocess text
    case_text_proc = h.preprocess_text(text_str=case_text)

    # Access preprocessed windows of consecutive bigrams
    windows = case_text_proc["bigram_windows"]
    words = case_text_proc["cleaned"]

    # Total number of words (to normalize scores)
    totalwords = len(words)

    ### A) Score unconditional scores
    risk = len([word for word in words if word in allwords["risk"]])
    sentpos = len([word for word in words if word in allwords["positive"]])
    sentneg = len([word for word in words if word in allwords["negative"]])
    covid = len([word for word in words if word in sarscov2_words])

    # Collect and prepare for conditional scores
    scores = {
        "Risk": risk,
        "Sentiment": sentpos - sentneg,
        "Covid": covid,
        "Pol": 0,
        "PRisk": 0,
        "PSentiment": 0,
        "Total words": totalwords,
    }

    ### B) Score conditional scores
    # Loop through each windows
    for window in windows:

        # Find middle ngram and check whether a "political" bigram
        middle_bigram = window[10]
        if middle_bigram not in political_bigrams:
            continue
        tfidf = political_bigrams[middle_bigram]["tfidf"]

        # Create word list for easy and quick access
        window_words = set([y for x in window for y in x.split()])

        # If yes, check whether risk synonym in window
        conditional_risk = (
            len([word for word in window_words if word in allwords["risk"]]) > 0
        )

        # If yes, check whether positive or negative sentiment
        conditional_sentpos = len(
            [word for word in window_words if word in allwords["positive"]]
        )
        conditional_sentneg = len(
            [word for word in window_words if word in allwords["negative"]]
        )

        # Weigh by tfidf
        conditional_risk = conditional_risk * tfidf
        conditional_sentpos = conditional_sentpos * tfidf
        conditional_sentneg = conditional_sentneg * tfidf

        # Collect results
        scores["Pol"] += tfidf
        scores["PRisk"] += conditional_risk
        scores["PSentiment"] += conditional_sentpos - conditional_sentneg

    # Append scores to file dict and then add to scores_list
    file.update(scores)
    scores_list.append(file)

# Collect in dataframe
scores_df = pd.DataFrame(scores_list)

# Scale
toscale = [
    x
    for x in scores_df.columns
    if x not in ["Total words", "fname", "id", "opinion_num"]
]
for column in toscale:
    scores_df[column] = scores_df[column] * 100000 * (1 / scores_df["Total words"])

# Write
scores_df.to_csv(output_dir + output_fname + ".tsv", sep="\t", encoding="utf-8")
