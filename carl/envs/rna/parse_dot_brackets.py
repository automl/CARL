from pathlib import Path


def parse_dot_brackets(
    dataset, data_dir, target_structure_ids=None, target_structure_path=None
):
    """TODO
    Generate the targets for next epoch.

    Args:
        dataset: The name of the benchmark to use targets from.
        data_dir: The directory of the target structures.
        target_structure_ids: Use specific targets by ids.
        target path: Specify a path to the targets.

    Returns:
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

    return [data_path.read_text().rstrip() for data_path in target_paths]
