import argparse
from itertools import chain
from asreview.extensions import extensions, load_extension


class NemoEntryPoint:

    description = "NEMO for ASReview."
    extension_name = "asreview-nemo"

    @property
    def version(self):
        try:
            from asreviewcontrib.nemo._version import __version__
            return __version__
        except ImportError:
            return "unknown"

    def execute(self, argv):
        parser = argparse.ArgumentParser(prog="asreview nemo")
        subparsers = parser.add_subparsers(dest="command", help="Subcommands for nemo")

        cache_parser = subparsers.add_parser(
            "cache", help="Cache specified entry points"
        )
        cache_parser.add_argument(
            "model_names",
            nargs="+",
            type=str,
            help="Model names to cache (e.g., 'xgboost', 'sbert').",
        )

        subparsers.add_parser("cache-all", help="Cache all available entry points")

        args = parser.parse_args(argv)
        if args.command == "cache":
            self.cache(args.model_names)
        elif args.command == "cache-all":
            self.cache([model.name for model in self._get_all_models()])
        else:
            parser.print_help()

    def cache(self, model_names):
        for name in model_names:
            try:
                model_class = load_extension("models.feature_extraction", name)
                self._load_model(model_class)
            except ValueError:
                try:
                    model_class = load_extension("models.classifiers", name)
                    self._load_model(model_class)
                except ValueError:
                    print(f"Error: Model '{name}' not found.")

    def _get_all_models(self):
        feature_extractors = extensions("models.feature_extraction")
        classifiers = extensions("models.classifiers")
        return list(
            chain(
                [fe for fe in feature_extractors if "asreviewcontrib.nemo_models" in str(fe)],
                [cls for cls in classifiers if "asreviewcontrib.nemo_models" in str(cls)],
            )
        )

    def _load_model(self, model):
        try:
            model_instance = model()
            _ = model_instance._model
            print(f"Successfully loaded: {model.name}")
        except Exception as e:
            print(f"Error loading model '{model.name}': {e}")
