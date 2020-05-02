from argparse import ArgumentParser
import gensim


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", type=str, required=True, help="model path")
    parser.add_argument(
            "-o", type=str, required=True, help="output model path"
            )
    parser.add_argument(
            "-n", type=int, required=True, help="n use words"
            )
    args = parser.parse_args()

    model =\
        gensim.models.KeyedVectors.load_word2vec_format(args.m, binary=True)

    new_model =\
        gensim.models.keyedvectors.WordEmbeddingsKeyedVectors(
                model.vector_size)

    for word in model.vocab:

        if word.lower() in new_model.vocab:
            continue

        new_model.add(word.lower(), model.get_vector(word))
        if len(new_model.vocab) == args.n:
            break

    new_model.save(args.o)
