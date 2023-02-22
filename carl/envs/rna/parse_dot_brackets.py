# flake8: noqa: F401
# isort: skip_file
from pathlib import Path

from typing import List, Optional, Union, Generator


def parse_dot_brackets(
    dataset: str,
    data_dir: str,
    target_structure_ids: Optional[List[int]] = None,
    target_structure_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Generate the targets for next epoch.

    Parameters
    ----------
    dataset : str
        The name of the benchmark to use targets from
    data_dir : str
        The directory of the target structures.
    target_structure_ids : Optional[Union[List[int], int]], optional
        Use specific targets by ids., by default None
    target_structure_path : Optional[str], optional
        pecify a path to the targets., by default None

    Returns
    -------
    Generator[int]
        An epoch generator for the specified target structures.
    """
    if target_structure_path:
        target_paths = [target_structure_path]
    elif target_structure_ids:
        target_paths = [
            Path(data_dir, dataset, f"{id_}.rna") for id_ in target_structure_ids
        ]
    else:
        target_paths = list(Path(data_dir, dataset).glob("*.rna"))

    return [data_path.read_text().rstrip() for data_path in target_paths]  # type: ignore[union-attr]
