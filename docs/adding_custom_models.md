# Adding Custom Models
This document describes the steps needed to implement custom models through ASReview Dory. Each ASReview LAB component has different requirements in terms of their interface and functions that are required to be implemented. Below are descriptions for each type and sub-type of components and how to extend ASReview Dory with a custom model.

## Custom Feature Extractor
Feature extractors in ASReview Dory exist in three ways: `SentenceTransformers` compatible transformer-enocders, `Huggingface` compatible transformer-encoders, and others. Each subsequent method requires more steps than the one before.

### SentenceTransformers and Huggingface
Example: [SentenceTransformer Feature Extractors](https://github.com/asreview/asreview-dory/blob/main/examples/sentence_transformer_fe.py)

Example: [HugginFace Feature Extractors](https://github.com/asreview/asreview-dory/blob/main/examples/huggingface_fe.py)

#### Steps:
1. Create a class that inherits from `asreviewcontrib.dory.feature_extractors.SentenceTransformerPipeline`
2. Add the following class variables:
    - `name`: How this model will show up in the CLI, internal data records, and project exports.
    - `label`: The label used to identify this model in ASReview LAB interface.
    - `default_model_name`: The full `SentenceTransformer` compatible model name
3. Add the feature extractor through Python entry points in `pyproject.toml`
    - Under `[project.entry-points."asreview.models.feature_extractors"]` add the custom model to register it as a Python entry point.
    - E.g., for the [HugginFace Feature Extractors](https://github.com/asreview/asreview-dory/blob/main/examples/huggingface_fe.py) you should add `"qwen3-embedding-8b" = "asreviewcontrib.dory.feature_extractors.transformer_embeddings:Qwen3Embedding8B"`
4. Install ASReview Dory so that the custom model is registered correctly and useable in ASReview LAB using `pip install . --no-cache-dir`

### Others
Example: [Doc2Vec Feature Extractors](https://github.com/asreview/asreview-dory/blob/main/examples/other_fe.py)

The example shows the implementation of Doc2Vec from the [gensim](https://github.com/piskvorky/gensim?tab=readme-ov-file#documentation) package.

## Custom Classifier
Classifiers exist in many ways, but for neural networks Dory provides a wrapper to quickly implement a custom network. Any other classifier can be implemented similar to [XGBoost](https://github.com/asreview/asreview-dory/blob/main/asreviewcontrib/dory/classifiers/xgboost.py) and [AdaBoost](https://github.com/asreview/asreview-dory/blob/main/asreviewcontrib/dory/classifiers/adaboost.py)
### Neural Network
Example: [Neural Network Classifiers](https://github.com/asreview/asreview-dory/blob/main/examples/neural_net_clf.py)

Steps:
1. Create a class that inherits from `asreviewcontrib.dory.classifiers.SentenceTransformerPipeline`
2. Add the following class variables:
    - `name`: How this model will show up in the CLI, internal data records, and project exports.
    - `label`: The label used to identify this model in ASReview LAB interface.
3. Add the `def _build_nn_model(self, X, y):` function, in which you can define your model using [Keras](https://github.com/keras-team/keras) syntax.
4. Add the feature extractor through Python entry points in `pyproject.toml`
    - Under `[project.entry-points."asreview.models.classifiers"]` add the custom model to register it as a Python entry point.
    - E.g., for the [Neural Network Classifiers](https://github.com/asreview/asreview-dory/blob/main/examples/neural_net_clf.py) you should add `"nn-1-layer" = "asreviewcontrib.dory.classifiers.neural_networks:NN1LayerClassifier"`
5. Install ASReview Dory so that the custom model is registered correctly and useable in ASReview LAB using `pip install . --no-cache-dir`

### Others
For other classifiers, please take a look at how Dory implements [XGBoost](https://github.com/asreview/asreview-dory/blob/main/asreviewcontrib/dory/classifiers/xgboost.py) and [AdaBoost](https://github.com/asreview/asreview-dory/blob/main/asreviewcontrib/dory/classifiers/adaboost.py).
