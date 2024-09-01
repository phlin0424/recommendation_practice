from datareader.ml_10m_data import BaseData


def indices_mapper(
    input_data: list[BaseData], id_col_name: str, reverse=False
) -> tuple[dict[int, int], dict[int, int] | None]:
    """Create a index mapper and a reversed mapper (when specified).

    Args:
        input_data (list[BaseData]): _description_
        id_col_name (str): The column that is used to create the index mapper.
        reverse (bool, optional): specified when want to derived the reversed mapper. Defaults to False.

    Returns:
        tuple[dict[int, int], dict[int, int] | None]: _description_
    """
    # Extract user_id or movie_id from the input data into a unique array
    id_cols = [item.get(id_col_name) for item in input_data]

    # Map userId and movieId to indices
    mapper = {user_id: index for index, user_id in enumerate(list(set(id_cols)))}

    # Create the reversed indices mapper when reverse is specified:
    if reverse:
        inverse_mapper = {index: id_col for id_col, index in mapper.items()}
        return mapper, inverse_mapper

    return mapper, None


if __name__ == "__main__":
    import asyncio
    from datareader.ml_10m_data import IntegratedDatas

    intergared_data = asyncio.run(IntegratedDatas.from_db(user_num=10))

    print(
        indices_mapper(
            input_data=intergared_data.data,
            id_col_name="user_id",
            reverse=True,
        )
    )
