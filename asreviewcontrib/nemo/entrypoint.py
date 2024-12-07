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

        list_parser = subparsers.add_parser(
            "list", help="List all available entry points"
            )
        list_parser.add_argument(
            "--classifiers",
            action="store_true",
            help="List only classifiers",
        )
        list_parser.add_argument(
            "--feature-extractors",
            action="store_true",
            help="List only feature extractors",
        )

        args = parser.parse_args(argv)
        if args.command == "cache":
            self.cache(args.model_names)
        elif args.command == "cache-all":
            self.cache([model.name for model in _get_all_models()])
        elif args.command == "list":
            model_type = (
                "fe" if args.feature_extractors else "cl" if args.classifiers else None
            )
            print([model.name for model in _get_all_models(model_type)])
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

    def _load_model(self, model):
        print(f"Loading {model.name}...")
        try:
            model_instance = model()
            _ = model_instance._model
        except Exception as e:
            print(f"Error loading model '{model.name}': {e}")


def _get_all_models(model_type=None):
    feature_extractors = extensions("models.feature_extraction")
    classifiers = extensions("models.classifiers")

    if model_type == "fe":
        models = list(
            {str(entry_point): entry_point for entry_point in feature_extractors 
             if "asreviewcontrib.nemo_models" in str(entry_point)}.values()
        )
    elif model_type == "cl":
        models = list(
            {str(entry_point): entry_point for entry_point in classifiers 
             if "asreviewcontrib.nemo_models" in str(entry_point)}.values()
        )
    else:
        models = list(
            {str(entry_point): entry_point for entry_point in chain(
                feature_extractors, classifiers
            ) if "asreviewcontrib.nemo_models" in str(entry_point)}.values()
        )

    return models
