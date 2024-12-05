import argparse
import pkg_resources
from asreview.entry_points import BaseEntryPoint


class NemoEntryPoint(BaseEntryPoint):
    description = "Nemo functionality for ASReview."
    extension_name = "asreview-nemo"

    @property
    def version(self):
        try:
            from asreviewcontrib.models._version import __version__
            return __version__
        except ImportError:
            return "unknown"

    def execute(self, argv):
        # Argument parser for the `nemo` command
        parser = argparse.ArgumentParser(prog="asreview nemo")
        subparsers = parser.add_subparsers(dest="command", help="Subcommands for nemo")

        # Subcommand: cache
        cache_parser = subparsers.add_parser("cache", help="Cache an entrypoint")
        cache_parser.add_argument(
            "models",
            nargs="+",
            type=str,
            help="Specify one or more entry points to load, e.g., 'xgboost sbert'.",
        )

        # Subcommand: cache-all
        subparsers.add_parser("cache-all", help="Cache all entry points")

        # Parse arguments
        args = parser.parse_args(argv)

        if args.command == "cache":
            self.cache(args.file)
        elif args.command == "cache-all":
            self.cache_all()
        else:
            parser.print_help()

    def cache(self, file_name):
        print(f"cache {file_name}")
        print(f"also known as {self._get_entry_point(file_name)}")
        # try:
        #     entry_point = self._get_entry_point(file_name)
        #     if entry_point:
        #         module = entry_point.load()
        #         print(f"Successfully loaded: {module.__name__}")
        #     else:
        #         print(f"Error: Entry point '{file_name}' not found.")
        # except ModuleNotFoundError as e:
        #     print(f"Error: Could not load the file '{file_name}'. {str(e)}")

    def cache_all(self):
        print(f"cache {self._get_all_entry_points}")
        # entry_points = self._get_all_entry_points()
        # for ep_name, entry_point in entry_points.items():
        #     try:
        #         module = entry_point.load()
        #         print(f"Successfully loaded: {module.__name__}")
        #     except Exception as e:
        #         print(f"Error loading '{ep_name}': {str(e)}")

    def _get_all_entry_points(self):
        """
        Retrieve all entry points defined under the classifiers and feature extraction 
        categories.
        """
        classifiers = pkg_resources.iter_entry_points("asreview.models.classifiers")
        feature_extraction = pkg_resources.iter_entry_points("asreview.models.feature_extraction")

        entry_points = {}
        for ep in classifiers:
            entry_points[ep.name] = ep
        for ep in feature_extraction:
            entry_points[ep.name] = ep

        return entry_points

    def _get_entry_point(self, name):
        """
        Retrieve a specific entry point by name.
        """
        entry_points = self._get_all_entry_points()
        return entry_points.get(name)
