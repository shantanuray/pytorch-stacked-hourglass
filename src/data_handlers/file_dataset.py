"""Interface to local file."""
import os
import gzip
import json


class FileDataSet():
    """Get images from local/network file system."""

    def __init__(self, image_folder):
        """Init local file connection."""
        self.image_folder = os.path.abspath(image_folder)
        self.annotations = os.path.abspath()

    def get_imageset(self):
        """List all images in an imageset."""
        image_list = os.listdir(self.image_folder)
        imageset = []
        for idx, value in enumerate(image_list):
            imageset[idx] = {"id": idx,
                             "path": os.path.join(image_list, value)}
        return(imageset)

    def get_image(self, image_name):
        """Get image from an folder."""
        image = os.path.join(self.img_folder, image_name)
        return(image)

    def get_annotations(self, annotation_source):
        """Get all annotations from gzipped file."""
        with gzip.open(os.path.abspath(annotation_source)) as f:
            return(json.load(f))
