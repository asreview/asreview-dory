# ASReview Dory 🐟

ASReview Dory is an extension to the [ASReview
software](https://github.com/asreview/asreview), providing new models for
classification and feature extraction. The extension is maintained by the
ASReview LAB team.

## Installation

You can install ASReview Dory via PyPI using the following command:

```bash
pip install asreview-dory
```

> ⚠️ **XGBoost on MacOS**
> If you are using macOS and plan to use XGBoost, you should first install OpenMP (`brew install libomp`)

## Model components

Feature Extractors:

    GTR T5
    LaBSE
    MPNet
    Multilingual E5
    MXBAI

Classifiers:

    AdaBoost
    Neural Network - 2-Layer
    Neural Network - Dynamic
    Neural Network - Warm Start
    XGBoost

Explore the performance of these models in our [Simulation
Gallery](https://jteijema.github.io/synergy-simulations-website/models.html)!
Look for the 🐟 icon to spot the Dory models.

## Usage

Once installed, the plugins will be available in the front-end of ASReview, as
well as being accessible via the command-line interface.

You can check all available models using:
```console
asreview algorithms
```

### Caching Models

You can pre-load models to avoid downloading them during runtime by using the
`cache` command. To `cache` specific models, such as `xgboost` and `sbert`, run:

```console
asreview dory cache nb xgboost sbert
```

To cache all available models at once, use:

```console
asreview dory cache-all
```

## Compatibility

This plugin is compatible with ASReview version 2 or later. Ensure that your ASReview
installation is up-to-date to avoid compatibility issues.

The development of this plugin is done in parallel with the development of the
ASReview software. We aim to maintain compatibility with the latest version of
ASReview, but please report any issues you encounter.

## Contributing

We welcome contributions from the community. To contribute, please follow these
steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Implement your changes.
4. Commit your changes with a clear message.
5. Push your changes to your fork.
6. Open a pull request to the main repository.

Please ensure your code adheres to the existing style and includes relevant
tests.

For any questions or further assistance, feel free to contact the ASReview Lab
Developers.

---

Enjoy using ASReview Dory! We hope these new models enhance your systematic
review processes.
