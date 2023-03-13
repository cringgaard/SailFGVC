import datasets
from datasets import Dataset, DatasetDict , ClassLabel

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
    "Hull Type" : ClassLabel(names_file="https://huggingface.co/datasets/cringgaard/boats_dataset/raw/main/labels/Hull_Type.txt"),
    "Rigging Type" : ClassLabel(names_file="https://huggingface.co/datasets/cringgaard/boats_dataset/raw/main/labels/Rigging_Type.txt"),
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
    "Construction" : ClassLabel(names_file="https://huggingface.co/datasets/cringgaard/boats_dataset/raw/main/labels/Construction.txt"),
    "Ballast Type" : ClassLabel(names_file="https://huggingface.co/datasets/cringgaard/boats_dataset/raw/main/labels/Ballast Type.txt"),
    # "First Built" : datasets.Value('float64'),
    # "# Built" : datasets.Value('float64'),
    "Designer" : ClassLabel(names_file="https://huggingface.co/datasets/cringgaard/boats_dataset/raw/main/labels/Designer.txt"),
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
    "name" : ClassLabel(names_file="https://huggingface.co/datasets/cringgaard/boats_dataset/raw/main/labels/name.txt"),
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

_IMAGES_DIR = "data/images"

_URL = "https://huggingface.co/datasets/cringgaard/boats_dataset/resolve/main/boats.tar.gz"


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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={
                    "filepath": "boat_data_train.csv",
                    "images": dl_manager.iter_archive(archive_path),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={
                    "filepath": "boat_data_test.csv",
                    "images": dl_manager.iter_archive(archive_path),
                }
            ),
        ]

    def _generate_examples(self, images, filepath):
        """This function returns the examples in the raw (text) form."""
        df = pd.read_csv(filepath)
        for file_path, file_obj in images:
            if sum(df['img_path'] == file_path.split("/")[1]) > 0:
                yield file_path, {
                    "img_path" : {"path": file_path, "bytes": file_obj.read()},
                    "Hull Type" : df['Hull Type'][df['img_path'] == file_path.split("/")[1]].values[0],
                    "Rigging Type" : df['Rigging Type'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "LOA" : df['LOA'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "LWL" : df['LWL'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Beam" : df['Beam'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "S.A. (reported)" : df['S.A. (reported)'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Draft (max)" : df['Draft (max)'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Displacement" : df['Displacement'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Ballast" : df['Ballast'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "S.A./Disp." : df['S.A./Disp.'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Bal./Disp." : df['Bal./Disp.'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Disp./Len." : df['Disp./Len.'][df['img_path'] == file_path.split("/")[1]].values[0],
                    "Construction" : df['Construction'][df['img_path'] == file_path.split("/")[1]].values[0],
                    "Ballast Type" : df['Ballast Type'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "First Built" : df['First Built'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "# Built" : df['# Built'][df['img_path'] == file_path.split("/")[1]].values[0],
                    "Designer" : df['Designer'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Comfort Ratio" : df['Comfort Ratio'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Capsize Screening Formula" : df['Capsize Screening Formula'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "S#" : df['S#'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "I" : df['I'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "J" : df['J'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "P" : df['P'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "E" : df['E'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "SPL/TPS" : df['SPL/TPS'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "ISP" : df['ISP'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "S.A. Fore" : df['S.A. Fore'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "S.A. Main" : df['S.A. Main'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "S.A./Disp. (calc.)" : df['S.A./Disp. (calc.)'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Est. Forestay Len." : df['Est. Forestay Len.'][df['img_path'] == file_path.split("/")[1]].values[0],
                    "name" : df['name'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Draft (min)" : df['Draft (min)'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Builder" : df['Builder'][df['img_path'] == file_path.split("/")[1]].values[0], 
                    # "Model" : df['Model'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Last Built" : df['Last Built'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Make" : df['Make'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Type" : df['Type'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "HP" : df['HP'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Fuel" : df['Fuel'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Headroom" : df['Headroom'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Water" : df['Water'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Website" : df['Website'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "KSP" : df['KSP'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "PY" : df['PY'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "EY" : df['EY'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Bridgedeck Clearance" : df['Bridgedeck Clearance'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Mast Height from DWL" : df['Mast Height from DWL'][df['img_path'] == file_path.split("/")[1]].values[0],
                    # "Bruce Number" : df['Bruce Number'][df['img_path'] == file_path.split("/")[1]].values[0],
                }
