# import sys
# import traceback

# from typing import Any, Callable
# from logs.logger import Logger

# def run_main(f: Callable[[], Any]) -> None:
#     """
#     Standard test runner with proper cleanup
#     """
#     try: f()
    
#     except KeyboardInterrupt:
#         print("\nAborted by user")
#         Logger.shutdown()
#         sys.exit(0)
    
#     except Exception as e:
#         print(f"Test failed: {e}")
#         traceback.print_exc()
#         Logger.shutdown()
#         sys.exit(1)
    
#     finally: Logger.shutdown()

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
