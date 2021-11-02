import torch
import clip
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
import cv2 as cv
import itertools
import copy

from nebula_api.milvus_api import MilvusAPI
from milvus import Milvus


def ret_sent(embedding_array, sentences, num_sent=1):
    single_threshold_list = []
    embedding_as_list = embedding_array[0, :].tolist()
    # Take one vector
    search_domain = sentences.search_vector(num_sent, embedding_as_list)
    # For loop is needed if we have two sentences or more
    for v in search_domain:
        single_threshold_list.append(v[1]['sentence'])
        print(v[0], v[1]['sentence'])

    max_score = 0
    best_sent = ''
    for k in range(100):
        embedding_as_list = (embedding_array[0, :] + 0.1*np.random.randn(1, 640))[0, :].tolist()
        search_domain = sentences.search_vector(num_sent, embedding_as_list)
        for v in search_domain:
            if v[0] > max_score:
                max_sore = v[0]
                best_sent = v[1]['sentence']

    return best_sent


def choose_best_setence_from_array(embedding_array, sentences):
    best_sent = ''
    best_dist = 0
    best_emb = embedding_array[0, :]
    for mmm in range(embedding_array.shape[0]):
        embedding_as_list = embedding_array[mmm, :].tolist()
        # Take one vector
        search_domain = sentences.search_vector(1, embedding_as_list)
        # For loop is needed if we have two sentences or more
        for v in search_domain:
            if v[0] > best_dist:
                best_dist = v[0]
                best_emb = embedding_array[mmm, :]
                best_sent = v[1]['sentence']

    return best_sent, best_emb


class NebulaVideoEvaluation:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # self.model1, self.preprocess1 = clip.load("RN50", device=self.device)
        # self.model2, self.preprocess2 = clip.load("RN101", device=self.device)
        self.model, self.preprocess = clip.load("RN50x4", device=self.device)
        self.model_res = 640


    def mark_blurred_frames(self, movie_name, start_frame, end_frame, blur_threshold=100):
        """
        :param movie_name: the full path
        :param start_frame: we are testing the movie only between frames start_frame and end_frame
        :param end_frame:
        :return: return the list of 1s and 0s and from start_frame (including) until end_frame (including)
        0 == BAD
        """
        cap = cv.VideoCapture(movie_name)
        ret, frame = cap.read()
        frame_num = 0
        ret_list = []
        values = []
        while cap.isOpened() and ret:
            if frame_num >= start_frame and frame_num <= end_frame:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                dst = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
                fm = cv.Laplacian(dst, cv.CV_64F).var()
                values.append(fm)
                if fm > blur_threshold:
                    ret_list.append(1)
                else:
                    ret_list.append(0)
            frame_num = frame_num + 1
            ret, frame = cap.read()

        blur_threshold = np.median(values) / 1.25

        cap = cv.VideoCapture(movie_name)
        ret, frame = cap.read()
        frame_num = 0
        ret_list = []
        values = []
        while cap.isOpened() and ret:
            if frame_num >= start_frame and frame_num <= end_frame:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                dst = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
                fm = cv.Laplacian(dst, cv.CV_64F).var()
                values.append(fm)
                if fm > blur_threshold:
                    ret_list.append(1)
                else:
                    ret_list.append(0)
            frame_num = frame_num + 1
            ret, frame = cap.read()

        return np.array(ret_list)

    def frame_to_sentence(self, frame):
        """
        Given an RGB frame, get sentences from it
        :param frame:
        :return:
        """
        pass

    def get_embedding_difs(self, movie_name, start_time, end_time):

        cap = cv.VideoCapture(movie_name)
        init = 0
        old_embeddings = None
        diff_list = []
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        embedding_array = np.zeros((0, self.model_res))
        frame_num = 0
        fps = cap.get(cv.CAP_PROP_FPS)
        while cap.isOpened() and ret:
            if (frame_num >= start_time * fps) and (frame_num < end_time * fps):
                with torch.no_grad():
                    img = self.preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)
                    embeddings = self.model.encode_image(img)
                    embeddings = embeddings / np.linalg.norm(embeddings)
                    embedding_array = np.append(embedding_array, embeddings, axis=0)
                    if init == 0:
                        init = 1
                    else:
                        diff = embeddings - old_embeddings
                        diff_list.append((diff * diff).sum())
                    old_embeddings = embeddings
            frame_num = frame_num + 1
            ret, frame = cap.read()
            if frame is not None:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        return embedding_array, fps


    def encode_text(self, text):
        text_token = torch.cat([clip.tokenize(text)]).to('cpu')
        return self.model.encode_text(text_token).detach().numpy()


    def create_clip_representation(self, movie_name, thresholds, start_time=-1, end_time=10000000, method='median') -> list:
        # method='median', 'average', 'single'
        # image = self.preprocess(Image.open("CLIP.png")).unsqueeze(0).to(self.device)
        # sentences = MilvusAPI('lables', 'nebula_dev')

        embedding_array, fps = self.get_embedding_difs(movie_name, start_time, end_time)

        start_frame = start_time * fps
        end_frame = end_time * fps
        blurred = self.mark_blurred_frames(movie_name, start_frame, end_frame, blur_threshold=40)

        boundaries = []
        good_frame_per = []
        embedding_list = []
        if embedding_array.shape[0] == 0:
            return embedding_list, boundaries
        for th in thresholds:
            new_bounds, good_frame_len = self.segment_embedding_array(embedding_array, th)
            good_frame_per.append(good_frame_len / (embedding_array.shape[0]))
            boundaries.append(new_bounds)
            avg_embeddings = np.zeros((0, self.model_res))
            for bound_tuple in new_bounds:
                good_ind = np.where(blurred[bound_tuple[0]:bound_tuple[1] + 1] == 1)[0] + bound_tuple[0]
                if len(good_ind) == 0:
                    continue
                if method == 'median':
                    represent_emb = np.median(embedding_array[good_ind, :], axis=0).reshape(1, self.model_res)
                elif method == 'average':
                    represent_emb = np.mean(embedding_array[good_ind, :], axis=0).reshape(1, self.model_res)
                elif method == 'single':
                    # right_ind = int((bound_tuple[0] + bound_tuple[1] + 1) / 2)
                    right_ind = good_ind[int(len(good_ind) / 2)]
                    represent_emb = (embedding_array[right_ind, :]).reshape(1, self.model_res)
                else:
                    return None, None


                avg_embeddings = np.append(avg_embeddings, represent_emb, axis=0)
                # _, represent_emb = choose_best_setence_from_array(embedding_array[bound_tuple[0]:bound_tuple[1] + 1, :],
                #                                                   sentences)

                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #
                # max_ind = np.argmax(np.array(dist[bound_tuple[0]:bound_tuple[1] + 1]))
                # represent_emb = embedding_array[max_ind + bound_tuple[0]].reshape(1, self.model_res)
                # avg_embeddings = np.append(avg_embeddings, represent_emb.reshape(1, self.model_res), axis=0)
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #
            embedding_list.append(avg_embeddings)
        pass
        # sch.linkage(embedding_array, method='ward')

        return embedding_list, boundaries

    def find_best_score_with_dynamic(self, diff_matrix) -> float:
        """
        Simple dynamic programmiing to find the best path
        :param diff_matrix:
        :return:
        """

        # direction 1
        cur_diff_matrix = copy.deepcopy(diff_matrix)

        for x in range(diff_matrix.shape[1] - 1):
            # check finish options for each row
            # and update the next column
            old_next_col = copy.deepcopy(cur_diff_matrix[:, x + 1])
            for y in range(diff_matrix.shape[0]):
                # at the same level, there is nothing to update
                # Try to update all higher levels
                for yy in range(y + 1, diff_matrix.shape[0]):
                    cur_diff_matrix[yy, x + 1] = max(cur_diff_matrix[y, x] + old_next_col[yy],
                                                     cur_diff_matrix[yy, x + 1])
        # choose the best score at the lowesst row
        best_row_score = np.max(cur_diff_matrix[-1, :])
        best_col_score = np.max(cur_diff_matrix[:, -1])
        best_score = max(best_col_score, best_row_score)

        return best_score / diff_matrix.shape[0]



    def find_best_score(self, diff_matrix) -> float:
        """
        The problem - there are two sequences of different length. The members of the sequences are called letters.
        There is a similarity of each letter to all other letters of the second sequence
        The goal is to find the similarity of sequences.
        We do it symmetrically in two directions.
        :param diff_matrix:
        :return:
        """

        # Start with vertical to horizontal
        # Go over all possible cases - from using all vertical letters to using only one vertical letter
        vertical_list = np.linspace(0, diff_matrix.shape[0] - 1, diff_matrix.shape[0]).astype('int32')
        horizontal_list = np.linspace(0, diff_matrix.shape[1] - 1, diff_matrix.shape[1]).astype('int32')

        best_sore = 0

        for k in range(1, min(diff_matrix.shape) + 1):
            vertical_options = list(itertools.combinations(vertical_list, k))
            horizontal_options = list(itertools.combinations(horizontal_list, k))

            k_coef = 0.5 * (k / diff_matrix.shape[0] + k / diff_matrix.shape[1])
            for vert in vertical_options:
                for hor in horizontal_options:
                    avg_score = np.mean(diff_matrix[vert, hor]) * k_coef
                    if avg_score > best_sore:
                        best_sore = avg_score

        return best_sore

        pass

    def find_similarity(self, embeedding_list0, embeedding_list1):
        # A sort of dynamic programming

        sim_score = 0
        for k in range(len(embeedding_list0)):
            pass

            # Create matrix N0 x M0 and put the
            diff_matrix = np.zeros((embeedding_list0[k].shape[0], embeedding_list1[k].shape[0]))
            for y in range(embeedding_list0[k].shape[0]):
                yv = embeedding_list0[k][y, :]
                yv = yv / np.linalg.norm(yv)
                for x in range(embeedding_list1[k].shape[0]):
                    xv = embeedding_list1[k][x, :]
                    xv = xv / np.linalg.norm(xv)

                    diff_matrix[y, x] = np.dot(xv, yv)

            # score = self.find_best_score(diff_matrix)
            score1 = self.find_best_score_with_dynamic(diff_matrix)
            score2 = self.find_best_score_with_dynamic(diff_matrix.T)
            score = 0.5*(score1 + score2)
            if score > sim_score:
                sim_score = score

        return sim_score


    def segment_embedding_array(self, embedding_array, dist_th) -> list:
        """
        option 1 - greedy, going from left to right, and stopping when the variance of the embeddings becomes too large
        or when the largest distance from the tested embedding to any embedding in the test is too large
        :return:
        """
        boundaries = []
        boundaries.append(0)
        for k in range(embedding_array.shape[0]):
            pass
            # compute distance from the current point to all other points
            max_dist = 10
            for m in range(boundaries[-1], k):
                d = np.sum(embedding_array[k]*embedding_array[m])
                # d = np.linalg.norm(embedding_array[k] - embedding_array[m])
                if d < max_dist:
                    max_dist = d
            if max_dist < dist_th:
                boundaries.append(k)
        boundaries.append(embedding_array.shape[0] - 1)
        # Remove very short shots
        # Create list of tuples
        ret_boundaries = []
        good_frame_len = 0
        for k in range(1, len(boundaries)):
            if boundaries[k] - boundaries[k - 1] > 9:
                ret_boundaries.append((boundaries[k - 1], boundaries[k]))
                good_frame_len = good_frame_len + boundaries[k] - boundaries[k - 1]

        return ret_boundaries, good_frame_len



