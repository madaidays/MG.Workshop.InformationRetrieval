import numpy as np
import pandas as pd
import os
import random


class IRModel():
    def __init__(self, inception, cards_csv_file_path, image_vectors_path, max_vectors=None):
        self.inception = inception
        self.max_vectors = max_vectors
        self.cards = pd.read_csv(cards_csv_file_path, sep=';')
        self.image_vectors = self.__get_image_vectors(image_vectors_path)

    def get_similar_card(self, image_path):
        img_tensor_val = self.inception.get_image_tensor(
            image_path)
        image_id = self.__get_closest_captions(
            img_tensor_val, self.image_vectors)
        card = self.cards[self.cards['id'] == image_id]
        return card.to_dict('records')[0]

    def __get_closest_captions(self, img_tensor_val, image_vectors):
        min_dist = 1000
        image_id = ''
        for vector in image_vectors:
            if np.linalg.norm(img_tensor_val - vector['vector']) < min_dist:
                min_dist = np.linalg.norm(
                    img_tensor_val - vector['vector'])
                image_id = vector['id']
        return image_id

    def __get_image_vectors(self, image_vectors_path):
        vectors_list = os.listdir(
            image_vectors_path) if not self.max_vectors else random.sample(os.listdir(
                image_vectors_path), self.max_vectors)
        return list(map(lambda vector: {'id': vector.replace('.jpg.npy', ''), 'vector': np.array([np.load(os.path.join(image_vectors_path, vector), allow_pickle=True)])},
                        vectors_list))
