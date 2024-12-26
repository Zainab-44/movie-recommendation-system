{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fe9acc48",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7219f340",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6fe5f487",
   "metadata": {},
   "source": [
    "# Load The Data Set Movies & Credits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9ab6680",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies = pd.read_csv(\"tmdb_5000_movies.csv\")\n",
    "credits = pd.read_csv(\"tmdb_5000_credits.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "17f3e3fd",
   "metadata": {},
   "source": [
    "# Explore The Data Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75199e73",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f45eaeff",
   "metadata": {},
   "outputs": [],
   "source": [
    "credits.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "196db8f3",
   "metadata": {},
   "source": [
    "As wee see in the movies dataset there is all the important information about movies like when movies release and in which language also about generes and budgets of the movies so we will keep some of these columns and skip some columns that we dont need.\n",
    "\n",
    ".................\n",
    "\n",
    "Similarly, in the credits dataset we have movies_id and information about cast of the movie and crew (all the other members in the movie) So, we have to datasets in the first step we need to be merged them so that we cxan easily handle only one dataframe for project, in the below stpe we will merge these two datasets."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e842476",
   "metadata": {},
   "source": [
    "# We are merging these two datasets on the basis of titles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d87219ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies = movies.merge(credits, on=\"title\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "74b35ead",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.shape\n",
    "\n",
    "#Now after merging the datasets we have 23 columns and 4809 rows"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2697c745",
   "metadata": {},
   "source": [
    "It seems that after merging the dataset we will have 24 columns, but we find it 23 This is because we meerged it on the basis odf title so it is not in the counting "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb9047ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "249bd59b",
   "metadata": {},
   "source": [
    "# Now lets Find out the columns that we need to be used "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dbdbe466",
   "metadata": {},
   "source": [
    "As we know we have to make a content based recommendation system so we are trying to create tags.\n",
    "\n",
    "here's the columns that are important and we will consider for our project\n",
    "\n",
    "1. genres\n",
    "2. movie_id\n",
    "3. keywords\n",
    "4. title\n",
    "5. overview\n",
    "6. cast (based on character like Salman khan etc)\n",
    "7. crew (director etc)\n",
    "\n",
    "now we will extract these columns for further proceed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2605b0f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies = movies[[\"movie_id\",\"title\",\"overview\",\"genres\",\"keywords\", \"cast\",\"crew\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3737bb0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "032df158",
   "metadata": {},
   "source": [
    "now we will create a new column TAGS, and we will combine overivew, generes, keywords, cast and crew to get tags column in this way we will have 3 columns \n",
    "\n",
    "1. movie_id\n",
    "2. title\n",
    "3. tags"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6294cd87",
   "metadata": {},
   "source": [
    "# Data Preprocessing "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d71346c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# handle missing values\n",
    "\n",
    "movies.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3abf5275",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e179050",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4f3999c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Checking the duplicate data\n",
    "\n",
    "movies.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ffe2bb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# chek the genere columns for merging to get tag colum\n",
    "\n",
    "movies.iloc[0].genres"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f561eb33",
   "metadata": {},
   "source": [
    "This is the list of dictionaries and our task is to get just this list \n",
    "\n",
    "[\"Action\",\"Adventure\",\"Fantasy\",\"Science Fiction\"]\n",
    "\n",
    "for this will make a function that will get this work done "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ac7b1181",
   "metadata": {},
   "outputs": [],
   "source": [
    "import ast\n",
    "def convert(obj):\n",
    "    L = []\n",
    "    for i in ast.literal_eval(obj):        \n",
    "        L.append(i[\"name\"])\n",
    "    return L"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "990d94af",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies[\"genres\"].apply(convert)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bda0746c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a53f6f52",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "70dca034",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies['genres'] = movies['genres'].apply(convert)\n",
    "movies.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bfc8b8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# same for keyword colums"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "70c59db1",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies['keywords'] = movies['keywords'].apply(convert)\n",
    "movies.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2e5c2ffd",
   "metadata": {},
   "outputs": [],
   "source": [
    "def convert3(obj):\n",
    "    L = []\n",
    "    counter = 0\n",
    "    for i in ast.literal_eval(obj):\n",
    "        if counter != 3:\n",
    "            L.append(i['name'])\n",
    "            counter+=1\n",
    "        else:\n",
    "            break\n",
    "    return L "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1de9a38",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies[\"cast\"] = movies[\"cast\"].apply(convert3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d7631afc",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a49ad33f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def fetch_director(obj):\n",
    "    L = []\n",
    "    for i in ast.literal_eval(obj):\n",
    "        if i['job'] == 'Director':\n",
    "            L.append(i['name'])\n",
    "            break\n",
    "    return L "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59a6228a",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies['crew'] = movies['crew'].apply(fetch_director)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e30b8e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84adce31",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies[\"overview\"] = movies[\"overview\"].apply(lambda x:x.split())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "861fce72",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b142484d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#remove the spaces in double words (one name) from these columns for avoiding conflit to get recomendation , becasue first names may be same for 2 persons \n",
    "\n",
    "movies[\"genres\"] = movies[\"genres\"].apply(lambda x:[i.replace(\" \",\"\") for i in x])\n",
    "movies[\"keywords\"] = movies[\"keywords\"].apply(lambda x:[i.replace(\" \",\"\") for i in x])\n",
    "movies[\"cast\"] = movies[\"cast\"].apply(lambda x:[i.replace(\" \",\"\") for i in x])\n",
    "movies[\"crew\"] = movies[\"crew\"].apply(lambda x:[i.replace(\" \",\"\") for i in x])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aec85d5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "69433239",
   "metadata": {},
   "source": [
    "# Concatenate the above columns to get tag column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "865a33e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "949a04fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "movies.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a57525d",
   "metadata": {},
   "source": [
    "As we acheived our task to make tag column now the other columns we will remove "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d59a5c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df = movies[[\"movie_id\", \"title\", \"tags\"]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f8c528e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6d7edd45",
   "metadata": {},
   "outputs": [],
   "source": [
    "#now convert the list of tags column(values) into string\n",
    "\n",
    "new_df[\"tags\"] = new_df[\"tags\"].apply(lambda x:\" \".join(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e405494d",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6387a4ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df[\"tags\"][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22d6046b",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df[\"tags\"] = new_df[\"tags\"].apply(lambda x:x.lower())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f63cad34",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a56c00c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18b8e4f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.stem.porter import PorterStemmer\n",
    "ps = PorterStemmer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ac4b5ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "def stem(text):\n",
    "    y = []\n",
    "    for i in text.split():\n",
    "        y.append(ps.stem(i))\n",
    "    return \" \".join(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "159afcc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df[\"tags\"] = new_df[\"tags\"].apply(stem)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d5b5bb7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "157d0fde",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e713e6f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "43ed2825",
   "metadata": {},
   "source": [
    "# Now Text vectorization"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "22528cc9",
   "metadata": {},
   "source": [
    "our task is to make a system that recommend the same type of movies based on tags, \n",
    "\n",
    "so our 1st task is to find out the similarity between the movies tags (paragraph) thats why we will use vectorization, we will apply technique bags of words on the tags columns so that we will have vector coresponding to each move and in this wahy we can get our target similar words for recommendation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46bf2c31",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "cv = CountVectorizer(max_features=5000,stop_words='english')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21367e30",
   "metadata": {},
   "outputs": [],
   "source": [
    "vectors = cv.fit_transform(new_df['tags']).toarray()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8149a83a",
   "metadata": {},
   "outputs": [],
   "source": [
    "vectors[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "369bee12",
   "metadata": {},
   "outputs": [],
   "source": [
    "cv.get_feature_names()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6db337b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics.pairwise import cosine_similarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9722c7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "similarity = cosine_similarity(vectors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5834b1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "similarity.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0ca78f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend(movie):\n",
    "    movie_index = new_df[new_df['title'] == movie].index[0]\n",
    "    distances = similarity[movie_index]\n",
    "    movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:6]\n",
    "    \n",
    "    for i in movies_list:\n",
    "        print(new_df.iloc[i[0]].title)\n",
    "       \n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b922b3d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "recommend(\"Avatar\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbe72014",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0acb311",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(new_df.to_dict(), open(\"movie_dict.pkl\",\"wb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c0e9dc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(similarity, open(\"similarity.pkl\",\"wb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9e898d1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
