"""
    tests / debug / parser.py
    -------------------------
   Argument parser for abstract the argument parser, it also include a 
   object pass `CommandSpec` and also an additional annotation @mainrunner to 
   avoid needed any logic in main within the main script for each test script
   instead all is included within the annotation.
"""
import argparse
import sys
import traceback

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List


@dataclass
class CommandSpec:
    name    : str
    help    : str
    handler : Callable
    args    : List[Dict[str,Any]]=field(default_factory=list)



def build_parser(commands: List[CommandSpec]) -> argparse.ArgumentParser:
    """
    """
    parser = argparse.ArgumentParser(description="Logger debug CLI")
    subparser = parser.add_subparsers(dest="command", required=True)

    for c in commands:
        p = subparser.add_parser(c.name, help=c.help)
        for a in c.args:
            p.add_argument(*a["flags"], **a["kwargs"])
        
        p.set_defaults(_handler=c.handler)
    
    return parser


def mainrunner(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args, **kwargs):
        try: return f(*args, **kwargs)
        
        except KeyboardInterrupt:
            print("\nAborted by user")
            raise
        
        except Exception as e:
            print(f"Test failed: {e}")
            traceback.print_exc()
            raise
    
    return wrapper