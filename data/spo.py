# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SPO"""

import json
import os

import datasets

_CITATION = """None"""

_DESCRIPTION = """SPO"""

_HOMEPAGE = "https://github.com/JunnYu/datasets_jy"

_LICENSE = "MIT"

_BASE_URL = "./"

_URLs = {
    "spo": _BASE_URL + "spo.zip",
}


class SPO(datasets.GeneratorBasedBuilder):
    """SPO"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="spo", version=VERSION, description="spo"),
    ]

    DEFAULT_CONFIG_NAME = "spo"

    def _info(self):

        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "spo_list": [
                    {
                        "predicate": datasets.Value(dtype="string"),
                        "object": datasets.Value(dtype="string"),
                        "subject": datasets.Value(dtype="string"),
                    }
                ],
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            # Here we define them above because they are different between the two configurations
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.name]

        data_dir = dl_manager.download_and_extract(my_urls)

        outputs = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train_data.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev_data.json"),
                    "split": "dev",
                },
            ),
        ]
        test_file = os.path.join(data_dir, "test_data.json")
        if os.path.exists(test_file):
            outputs.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "filepath": test_file,
                        "split": "test",
                    },
                )
            )
        return outputs

    def _generate_examples(
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
        self,
        filepath,
        split,
    ):
        """Yields examples as (key, example) tuples."""
        id_ = 0
        with open(filepath, "r", encoding="utf8") as f:
            for l in f:
                l = json.loads(l)
                yield id_, {
                    "id": str(id_),
                    "text": l["text"],
                    "spo_list": [
                        {
                            "subject": spo["subject"],
                            "predicate": spo["predicate"],
                            "object": spo["object"],
                        }
                        for spo in l["spo_list"]
                    ],
                }
                id_ += 1
