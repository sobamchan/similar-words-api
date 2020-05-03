import os
import random
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, Body, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gensim
import editdistance


load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:8080",
    "https://eigo-mimi.herokuapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH_KEY = os.getenv("SECRET")
model = gensim.models.KeyedVectors.load("./m.model")
vocab = [w for w, _ in model.vocab.items()]


def sim_word2vec(word, exclude_words=[]):
    for _w, _ in model.most_similar(word):
        if _w not in exclude_words:
            return _w
    return _w


def sim_editdistance(word, exclude_words=[]):
    best_distance = 1e+3
    best_word = 'HOGE'
    for v in vocab:
        if v in exclude_words:
            continue
        distance = editdistance.eval(word, v)
        if distance < best_distance:
            best_word = v
            best_distance = distance
    return best_word


@app.post("/get_similar_words")
def get(
        words: List[str] = Body(..., embed=True),
        n: int = Body(..., embed=True),
        Authorization: str = Header(None)
        ):

    if Authorization != AUTH_KEY:
        raise HTTPException(status_code=401, detail="wrong authorization key")

    list_sim_words = []

    for word in words:

        sim_words = []
        for _ in range(n):

            if word in vocab and random.random() > 0.7:
                # word2vec
                sim_word = sim_word2vec(word, sim_words + [word])
            else:
                # edit distance
                sim_word = sim_editdistance(word, sim_words + [word])

            sim_words.append(sim_word)

        list_sim_words.append(sim_words)

    return {"similar_words": list_sim_words}
