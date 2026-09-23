"""Written by component-belief: record what this process opens, under the project root."""
import atexit, os, sys

_ROOT = os.environ.get("BELIEF_READ_ROOT", "")
_OUT = os.environ.get("BELIEF_READS", "")
_seen = set()


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            path = os.path.realpath(os.fspath(args[0]))
        except (TypeError, ValueError):
            return
        if _ROOT and path.startswith(_ROOT):
            _seen.add(path)


def _flush():
    try:
        with open(_OUT, "a", encoding="utf-8") as fh:
            for path in sorted(_seen):
                fh.write(path + "\n")
    except OSError:
        pass


if _ROOT and _OUT:
    sys.addaudithook(_audit)
    atexit.register(_flush)

try:                              # keep any sitecustomize the environment already had
    import sitecustomize_original  # noqa: F401
except ImportError:
    pass
