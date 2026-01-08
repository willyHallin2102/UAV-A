import argparse

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class CommandSpec:
    name    : str
    help    : str
    handler : Callable
    extra_args  : List[Dict[str,Any]]=field(default_factory=list)



def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Adds simple commons and shared commands that all subparsers 
    are required to carry, such as all is required to share the
    logger instance passed.
    """
    parser.add_argument(
        "--loglevel", default="DEBUG", help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--to-disk",action="store_true",help="Enable disk logging")
    parser.add_argument("--no-json",action="store_true",help="Disable JSON format")



def build_parser(commands: List[CommandSpec]) -> argparse.ArgumentParser:
    """
    """
    parser = argparse.ArgumentParser(description="Logger debug CLI")
    subparser = parser.add_subparsers(dest="command", required=True)

    for command in commands:
        p = subparser.add_parser(command.name, help=command.help)
        add_common_arguments(p)

        for argument in command.extra_args:
            p.add_argument(*arg["flags"], **arg["kwargs"])
        
        # Bind handler directly (argparse-native, no lookup dict)
        p.set_defaults(_handler=command.handler)
    
    return parser
