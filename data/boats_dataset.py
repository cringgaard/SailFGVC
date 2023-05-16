import datasets
from datasets import Dataset, DatasetDict , ClassLabel
from .classes import *
import numpy as np

import os
from datasets.tasks import ImageClassification
import pandas as pd

cwd = os.getcwd()

logger = datasets.logging.get_logger(__name__)

_CITATION = ""

_DESCRIPTION = """\
FGVC of boats
"""
_FEATURES = datasets.Features({
    "img_path" : datasets.Image(),
    "Hull Type" : ClassLabel(names = list(Hull_Type_Classes.values())),
    "Rigging Type" : ClassLabel(names = list(Rigging_Type_Classes.values())),
    # "LOA" : datasets.Value('float64'),
    # "LWL" : datasets.Value('float64'),
    # "Beam" : datasets.Value('float64'),
    # "S.A. (reported)" : datasets.Value('float64'),
    # "Draft (max)" : datasets.Value('float64'),
    # "Displacement" : datasets.Value('float64'),
    # "Ballast" : datasets.Value('float64'),
    # "S.A./Disp." : datasets.Value('float64'),
    # "Bal./Disp." : datasets.Value('float64'),
    # "Disp./Len." : datasets.Value('float64'),
    "Construction" : ClassLabel(names = list(Construction_Classes.values())),
    "Ballast Type" : ClassLabel(names = list(Ballast_Type_Classes.values())),
    # "First Built" : datasets.Value('float64'),
    # "# Built" : datasets.Value('float64'),
    "Designer" : ClassLabel(names= list(Designer_Classes.values())),
    # "Comfort Ratio" : datasets.Value('float64'),
    # "Capsize Screening Formula" : datasets.Value('float64'),
    # "S#" : datasets.Value('float64'),
    # "I" : datasets.Value('float64'),
    # "J" : datasets.Value('float64'),
    # "P" : datasets.Value('float64'),
    # "E" : datasets.Value('float64'),
    # "SPL/TPS" : datasets.Value('float64'),
    # "ISP" : datasets.Value('float64'),
    # "S.A. Fore" : datasets.Value('float64'),
    # "S.A. Main" : datasets.Value('float64'),
    # "S.A./Disp. (calc.)" : datasets.Value('float64'),
    # "Est. Forestay Len." : datasets.Value('float64'),
    "name" : ClassLabel(names = list(Name_Classes.values())),
    # "Draft (min)" : datasets.Value('float64'),
    # "Builder" : datasets.Value('float'),
    # "Model" : datasets.Value('float'),
    # "Last Built" : datasets.Value('float64'),
    # "Make" : datasets.Value('float'),
    # "Type" : datasets.Value('float'),
    # "HP" : datasets.Value('float'),
    # "Fuel" : datasets.Value('float'),
    # "Headroom" : datasets.Value('float64'),
    # "Water" : datasets.Value('float'),
    # "Website" : datasets.Value('float'),
    # "KSP" : datasets.Value('float64'),
    # "PY" : datasets.Value('float64'),
    # "EY" : datasets.Value('float64'),
    # "Bridgedeck Clearance" : datasets.Value('float64'),
    # "Mast Height from DWL" : datasets.Value('float64'),
    # "Bruce Number" : datasets.Value('float64'),
    # "Path": datasets.Value("string"),
})

# _IMAGES_DIR = "data/images"

_URL = "https://huggingface.co/datasets/cringgaard/boats_dataset/resolve/main/boats.tar.gz"
_URL2 = "https://huggingface.co/datasets/cringgaard/boats_dataset/resolve/main/boat24.tar.gz"
_URL3 = "https://huggingface.co/datasets/cringgaard/boats_dataset/resolve/main/image_search.tar.gz"


# class BOATSConfig(datasets.BuilderConfig): #
#     """BuilderConfig for BOATS."""

#     def __init__(self, **kwargs):
#         """BuilderConfig for BOATS.
#         Args:
#           **kwargs: keyword arguments forwarded to super.
#         """
#         super(BOATSConfig, self).__init__(**kwargs)


class BOATS(datasets.GeneratorBasedBuilder):
    """BOATS"""

    # BUILDER_CONFIGS = [
    #     BOATSConfig(
    #         name="boats",
    #         version=datasets.Version("0.0.0", ""),
    #         description="BOATS",
    #     ),
    # ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            # supervised_keys=["Hull Type" , "Rigging Type"],
            homepage="https://huggingface.co/datasets/cringgaard/boats_dataset/tree/main",
            citation=_CITATION,
            license="",
            # task_templates=[ImageClassification(image_column="img_path", label_column="Hull Type")],

        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download(_URL)
        archive_path2 = dl_manager.download(_URL2)
        archive_path3 = dl_manager.download(_URL3)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={
                    "filepath": "https://huggingface.co/datasets/cringgaard/boats_dataset/raw/main/boat_data_train.csv",
                    "images": dl_manager.iter_archive(archive_path),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={
                    "filepath": "https://huggingface.co/datasets/cringgaard/boats_dataset/raw/main/boat_data_test.csv",
                    "images": dl_manager.iter_archive(archive_path),
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split("sailboatdata"), 
                gen_kwargs={
                    "filepath": "https://huggingface.co/datasets/cringgaard/boats_dataset/raw/main/boat_data_clean.csv",
                    "images": dl_manager.iter_archive(archive_path),
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split("boat24"), 
                gen_kwargs={
                    "filepath": "https://huggingface.co/datasets/cringgaard/boats_dataset/resolve/main/boat24_data_clean.csv",
                    "images": dl_manager.iter_archive(archive_path2),
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split("image_search"), 
                gen_kwargs={
                    "filepath": "https://huggingface.co/datasets/cringgaard/boats_dataset/resolve/main/image_search_data.csv",
                    "images": dl_manager.iter_archive(archive_path3),
                }
            ),
        ]

    def _generate_examples(self, images, filepath):
        """This function returns the examples in the raw (text) form."""
        df = pd.read_csv(filepath)
        for file_path, file_obj in images:
            if sum(df['img_path'] == file_path.split("/")[-1]) > 0:
                idx = df.index[(df['img_path'] == file_path.split("/")[-1])].values[0]
                yield file_path , {
                    "img_path"      : {"path": file_path, "bytes": file_obj.read()},
                    "Hull Type"     : Hull_Type_Classes[df['Hull Type'][idx]],
                    "Rigging Type"  : Rigging_Type_Classes[df['Rigging Type'][idx]],
                    "Construction"  : Construction_Classes[df['Construction'][idx]],
                    "Ballast Type"  : Ballast_Type_Classes[df['Ballast Type'][idx]],
                    "Designer"      : Designer_Classes[df['Designer'][idx]],
                    "name"          : Name_Classes[df['name'][idx]],
                }