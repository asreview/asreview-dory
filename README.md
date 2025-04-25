# ASReview NEMO: New Exciting Models

This repository is an expansion to the [ASReview
software](https://github.com/asreview/asreview), providing new models for
classification and feature extraction. It is maintained by the [ASReview Lab
Developers](https://asreview.ai/about).

You can install ASReview NEMO via PyPI using the following command:

```bash
pip install asreview-nemo
```

## Overview

ASReview NEMO (New Exciting Models) is a plugin for the ASReview software that
provides new and exciting classification and feature extraction models,
expanding the capabilities of ASReview for automated systematic reviews.

### Classifiers Available

    xgboost
    nn-2-layer
    dynamic-nn
    adaboost

### Feature Extractors Available

    sbert
    labse
    mxbai
    multilingual e5 large
    gtr t5 large

Explore the performance of these models in our [Simulation
Gallery](https://jteijema.github.io/synergy-simulations-website/models.html)!
Look for the 🐟 icon to spot the NEMO models.

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
asreview nemo cache nb xgboost sbert
```

To cache all available models at once, use:

```console
asreview nemo cache-all
```

## Compatibility

This plugin is compatible with `ASReview version 2+`. Ensure that your ASReview
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

Enjoy using ASReview NEMO! We hope these new models enhance your systematic
review processes.
