from datasets import Dataset, DatasetDict, ClassLabel


def stratified_split(dataset: Dataset, by_column: str, split_size: float = 0.5) -> DatasetDict:
    """
    Performs a stratified split on a dataset based on a specified column.

    Args:
        dataset (Dataset): The dataset to split.
        by_column (str): The column by which to stratify the split.
        split_size (float, optional): The proportion of data to be allocated to the split.
            Defaults to 0.5.

    Returns:
        DatasetDict: A DatasetDict containing two datasets: 'split' and 'remaining'.
    """
    # Get unique values in the specified column
    vals_in_col = list(set(dataset[by_column]))

    # Check if the column is not already a ClassLabel, if not, convert it
    if not isinstance(dataset.features.get(by_column), ClassLabel):
        dataset = dataset.cast_column(by_column, ClassLabel(names=vals_in_col))

    # Perform the stratified split and return the result as a DatasetDict
    return DatasetDict(
        zip(
            ("split", "remaining"),
            dataset.train_test_split(
                train_size=split_size, stratify_by_column=by_column
            ).values(),
        )
    )
