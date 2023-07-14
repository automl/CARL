# flake8: noqa: F401
# isort: skip_file
from email.generator import Generator
from pathlib import Path

from typing import List


def parse_dot_brackets(
    dataset: str,
    data_dir: str,
    target_structure_ids: List[int] = None,
    target_structure_path: Path = None,
) -> List[str]:
    """Generate the targets for next epoch.

    The most common encoding for the RNA secondary structure is the dot-bracket
    notation, consisting in a balanced parentheses string composed by a
    three-character alphabet {.,(,)}, that can be unambiguously converted
    in the RNA secondary structure.

    Parameters
    ----------
    dataset: str
        The name of the benchmark to use targets from
    data_dir: str
        The directory of the target structures.
    target_structure_ids: List[int]
        Use specific targets by ids.
    target_structure_path: str
        Specify a path to the targets

    Returns
    -------
    List[str]
        An epoch generator for the specified target structure(s)
    """

    if target_structure_path:
        target_paths = [target_structure_path]
    elif target_structure_ids:
        target_paths = [
            Path(data_dir, dataset, f"{id_}.rna") for id_ in target_structure_ids
        ]
    else:
        target_paths = list(Path(data_dir, dataset).glob("*.rna"))

    return [data_path.read_text().rstrip() for data_path in target_paths]
