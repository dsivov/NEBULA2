from common.manager import Manager
from common.manager import logging
from .google.google import Google
from common.cfg import Cfg
from .aws.aws import Aws
import os
from posixpath import basename
from experts.scene_detector_api import NEBULA_SCENE_DETECTOR
import boto3

class VTMaster(Manager):
    """
    This class is responsible for getting new videos, dividing them into scenes and sennding them to tracker
    """
    def __init__(self):
        self.scene_detector = NEBULA_SCENE_DETECTOR()
        self.storage_client = boto3.Session().client('s3')
        self.cfg = Cfg(['esdb'])
        self.bucket_name = self.cfg.get('esdb', 'upload_bucket_name')
        self.google = Google(logging, self.send_msg_self)
        self.last_movie_id = -1

    def process_new_movie(self, file_path):
        """
        The function is triggered by a new movie
        :param file_path: path to the new movie
        :return:
        The function divides the movie into scenes and saves them in our database
        """
        file_name = basename(file_path)
        movie_name = file_name.split('.')[0]
        self.last_movie_id = self.scene_detector.insert_movie(file_name, movie_name,
                                                              ["hollywood", "pegasus", "visual genome"], file_path,
                                                              "media/videos/" + file_name, 30000)

    def upload_file(self, file_path):
        """Upload file to S3"""
        # S3 can not have spaces
        file_name = basename(file_path)
        file_name = file_name.replace(' ', '_')
        dest_blob_name = 'media/videos/' + file_name
        blob = self.storage_client
        blob.upload_file(file_path, self.bucket_name, dest_blob_name)
        logging.debug('File {} uploaded to {}.'.format(
            file_path, dest_blob_name))
        return 's3://' + self.bucket_name + '/' + dest_blob_name

    def send_file_to_google(self, file_path):
        self.google.process_vtag(file_path, self.last_movie_id)