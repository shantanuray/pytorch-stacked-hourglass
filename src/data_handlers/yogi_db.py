"""Interface to Yogi DB."""
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from yogi.models import Image, ImageSet, Label, LabelSource


class YogiDB():
    """Access to a Yogi DB."""

    def __init__(self, database_url):
        """Init Yogi DB connection."""
        engine = create_engine(database_url)
        session = sessionmaker(bind=engine)
        self.session = session()

    def get_session(self):
        """Get session access directly."""
        return(self.session)

    def get_all(self, obj):
        """Get all rows from a given data object."""
        rows = self.session.query(obj).all()
        return(rows)

    def get_filtered(self, obj, **kwargs):
        """Get all rows from a given data object."""
        rows = self.session.query(obj).filter_by(**kwargs).all()
        return(rows)

    def get_imageset(self, image_set_name):
        """List all images in an imageset."""
        imageset = self.session.query(ImageSet).filter_by(name=image_set_name).one()
        return(imageset.images)

    def get_image(self, **kwargs):
        """Get an image from an imageset."""
        # Using kwargs will let us pass id, frame number - whatever we want
        image = self.session.query(Image).filter_by(**kwargs).one()
        return(image)

    def get_annotations(self, image_set_name="bigpaw",
                        annotation_source="basic-thresholder"):
        """Get annotations for a given image set."""
        imageset = self.session.query(ImageSet).filter_by(name=image_set_name).one()
        # Get session
        session = self.get_session()
        # Get the Image Set from the image_set_name
        imageset = self.get_filtered(ImageSet, name=image_set_name)
        # Get the Label Source
        label_source = self.get_filtered(LabelSource, name=annotation_source)
        """ Get annotations for the given ImageSet
        Use ImageSet.get_labelset(label_source, session)."""
        annotation_tuples = imageset[0].get_labelset(label_source[0], session)
        """ Convert annotations from tuple to json
        Data sructure:
            (dataset,
            image path,
            image width,
            image height,
            objpos, [annotation x coordinate, annotation y coordinate]
            joint_self: [[annotation x coordinate, annotation y coordinate,
                    annotation confidence]]."""
        annotations = []
        for i in annotation_tuples:
            if i[3] == 'hidden':
                x, y = -1, -1
                confidence = 0.0
            else:
                x, y = i[3], i[4]
                confidence = 1.0
            annotations.append({"dataset": image_set_name,
                                "image_path": i[0],
                                "img_width": i[1],
                                "img_height": i[2],
                                "objpos": [x * i[1], y * i[2]],
                                "scale_provided": 1.0,
                                "joint_self": [[x * i[1], y * i[2], confidence]]})
        return(annotations)
