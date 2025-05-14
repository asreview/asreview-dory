import argparse
from itertools import chain

from asreview.extensions import extensions, load_extension


class DoryEntryPoint:
    description = "Dory for ASReview."
    extension_name = "asreview-dory"

    @property
    def version(self):
        try:
            from asreviewcontrib.dory._version import __version__

            return __version__
        except ImportError:
            return "unknown"

    def execute(self, argv):
        parser = argparse.ArgumentParser(prog="asreview dory")
        subparsers = parser.add_subparsers(dest="command", help="Subcommands for Dory")

        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {self.version}",
            help="Show the version of the application",
        )

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
        subparsers.add_parser("list", help="List all available entry points")

        args = parser.parse_args(argv)
        if args.command == "cache":
            self.cache(args.model_names)
        elif args.command == "cache-all":
            self.cache([model.name for model in self._get_all_models()])
        elif args.command == "list":
            print([model.name for model in self._get_all_models()])
        else:
            parser.print_help()

    def cache(self, model_names) -> None:
        for name in model_names:
            try:
                entry_point = load_extension("models.feature_extractors", name)
                try:
                    # Try to load sentence-transformers
                    _ = (
                        entry_point(verbose=False)
                        .named_steps["sentence_transformer"]
                        ._model
                    )
                except KeyError:
                    pass
                print(f"Loaded FE {name}")
            except ValueError:
                try:
                    load_extension("models.classifiers", name)
                    print(f"Loaded CLS {name}")
                except ValueError:
                    print(f"Error: Model '{name}' not found.")

    def _get_all_models(self) -> list:
        feature_extractors = extensions("models.feature_extractors")
        classifiers = extensions("models.classifiers")
        return list(
            chain(
                [fe for fe in feature_extractors if "asreviewcontrib.dory" in str(fe)],
                [cls for cls in classifiers if "asreviewcontrib.dory" in str(cls)],
            )
        )
