from action_based_risk.time2vec.time2vec.time2vec import Time2Vec
import logging#
import os


class ConceptualEmbeddings:

    def __init__(self, concept_data_file="./actions_1.csv", train_file_name="./notebooks/train_concept_final.csv"):
        self.concept_df_path = concept_data_file
        self.train_file_name = train_file_name

    def train_embeddings(self):
        print(os.getcwd())
        self.model = Time2Vec(data_file_name=self.concept_df_path,
                 min_count=1,
                 subsampling=0,
                 train_file_name=self.train_file_name,
                 decay=1,
                 unit=2,
                 const=1000.,
                 rate=0.3,
                 limit=400,
                 chunk_size=1000,
                 processes=15,
                 dimen=50,
                 num_samples=2,
                 optimizer=2,
                 lr=1.0,
                 min_lr=0.1,
                 batch_size=16,
                 epochs=500,
                 valid=0,
                 seed=1)

        print("Model created...")

        self.model.build_vocab()
        #self.model.gen_train_data()
        self.model.learn_embeddings()
        self.model.save("./embeddings.pkl")



if __name__ == "__main__":
    conceptual_embeddings = ConceptualEmbeddings()

    conceptual_embeddings.train_embeddings()
