# Persian Movie Recommendation System — NLP

A text-search / recommendation system that takes a short Persian query
(e.g. `فیلم پلیسی` — "police movie") and returns the most relevant Persian movies
based on their descriptions.

It combines two classic NLP techniques:

1. **Word2Vec** — expands the query with related words.
2. **TF-IDF + cosine similarity** — ranks every movie description against the expanded query.

> Everything lives in one notebook: [`Recom_persian.ipynb`](Recom_persian.ipynb).
> The English version (resumes dataset) is in a separate repo, `recom-system-english`.

---

## How it works

```
query ──► clean ──► stem ──► Word2Vec: add similar words ──► TF-IDF vector ─┐
                                                                            ├─► cosine similarity ─► top-N movies
movie text ──► clean ──► stem ─────────────────────────────► TF-IDF matrix ─┘
```

### 1. Data loading
- `dataset.csv` — Persian movies (link, description in `Content_1`, `Genre`, …).
- `all_words.txt` — Persian stop words, a CSV with a `words` column (1,560 words,
  e.g. `را`, `باید`, `فیلم`, `کرد`).

### 2. Preprocessing
- Missing descriptions in `Content_1` are filled with `''`.
- English genre names are mapped to Persian words (`Crime → جرم و جنایت جنایی`, `Comedy → کمدی`, …).
- All columns are cast to `str`.
- The Persian genre is **appended to `Content_1`**, so genre words also count in the search.

### 3. Stop-word list = file words + most frequent corpus words
`CountVectorizer` counts every word in the corpus. The **top 30** most frequent words
(`به`, `می`, `در`, `که`, `از`, `را`, `است`, …) are appended to the stop-word list
→ **1,590 stop words** in total.

### 4. Cleaning — `remove_words_from_text(text)`
- Builds one regex pattern from all stop words (whole-word match, case-insensitive) and removes them.
- Removes special characters (`[^\w\s]`).

This regex approach is different from the English notebook's cleaner, which splits the text and filters
words instead (the regex version had problems there).

### 5. Stemming — `apply_hazm_stemmer(text)`
Uses the [`hazm`](https://github.com/roshan-research/hazm) `Stemmer` to reduce words to their roots,
and **drops words with 3 letters or fewer**.
Example: `یک فیلمی احساسی و پلیسی` → `فیلم احساس پلیس`.
Some stems may look odd, but that does not matter: every word goes through the same stemmer, so
the forms still match each other.

### 6. Query expansion — `word2vec(column, cleaner, user_input, num_results=10)`
1. Clean + stem the text column and tokenize it.
2. Train `Word2Vec` (`vector_size=150`, `window=4`, `min_count=10`, `workers=4`).
3. Clean + stem the user query.
4. For every query word longer than 3 characters, take its **5 most similar words**
   (scores are summed if a word shows up more than once).
5. Keep only words with a score **≥ 0.7**.
6. Return `"<query> <similar words…>"`.

Example: `فیلم پلیسی` → `پلیس همسر موقع موفق ایر خانواده`.

### 7. Ranking — `tf_idf(X_column, user_input, num_results=10)`
1. Build the TF-IDF matrix of the cleaned + stemmed descriptions.
2. Build the expanded query (original words + Word2Vec words, stemmed, short words dropped).
3. Transform the query into the same TF-IDF space.
4. Compute cosine similarity against every movie and return the top `num_results` descriptions.

The original query words appear twice in the expanded query, which gives them more weight than
the Word2Vec words.

---

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
jupyter notebook Recom_persian.ipynb
```

Put `dataset.csv` and `all_words.txt` next to the notebook (not included in this repo — see `.gitignore`), then:

```python
tf_idf(df['Content_1'], 'فیلم پلیسی')
```

## Notes & known limitations
- **Small dataset:** the Word2Vec similarity scores come out almost identical (~0.9997), so the
  expansion words are weak (`همسر`, `موقع`, `ایر` …). A bigger dataset gives much better results — see the
  English project.
- Word2Vec is **retrained on every call**; train once and reuse the model for real use.
- `TfidfVectorizer(stop_words='english')` has no effect on Persian text; stop words are already removed earlier.
- `tf_idf` returns rows from the global `df`, not from the `X_column` argument.
- `word2vec` and `tf_idf` print debug output (the raw similarity dict, the final query).
- `Lemmatizer` is imported but unused.
- Some genre translations in the mapping contain typos (e.g. `ععلایق`, `حیت وحش`) — worth fixing.
- The movie data comes from a third-party site (imvbox) — check permission before publishing it.
