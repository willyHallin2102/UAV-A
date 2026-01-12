import sys
import traceback

from functools import wraps
from typing import Callable


def mainrunner(f: Callable) -> Callable:
    """
    Decorator to wrap test-functions with proper logger cleaning
    """
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
