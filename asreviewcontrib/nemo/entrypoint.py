import argparse
from itertools import chain
from asreview.entry_points import BaseEntryPoint
from asreview.models.classifiers import list_classifiers
from asreview.models.feature_extraction import list_feature_extraction


class NemoEntryPoint(BaseEntryPoint):
    description = "Nemo functionality for ASReview."
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
            "model_name",
            nargs="+",
            type=str,
            help="Model names to cache (e.g., 'xgboost', 'sbert').",
        )

        subparsers.add_parser("cache-all", help="Cache all available entry points")

        args = parser.parse_args(argv)
        if args.command == "cache":
            self.cache(args.model_name)
        elif args.command == "cache-all":
            self.cache([model.__name__ for model in self._get_all_models()])
        else:
            parser.print_help()


    def cache(self, model_names):
        models = self._get_all_models()
        for name in model_names:
            model = next(
                (m for m in models if m.__name__.lower() == name.lower()), None
            )
            if model:
                self._load_model(model)
            else:
                print(f"Error: Model '{name}' not found.")


    def _get_all_models(self):
        classifiers = list_classifiers()
        feature_extractors = list_feature_extraction()
        return list(
            chain(
                [cls for cls in classifiers if "asreviewcontrib" in str(cls)],
                [fe for fe in feature_extractors if "asreviewcontrib" in str(fe)],
            )
        )


    def _load_model(self, model_class):
        try:
            model_instance = model_class()
            _ = model_instance._model
            print(f"Successfully loaded: {model_class.__name__}")
        except Exception as e:
            print(f"Error loading model '{model_class.__name__}': {e}")
