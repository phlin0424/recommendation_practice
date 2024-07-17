from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import TypeVar

# +++++++++++++++++++++++++++++++++++++++++++++++++
# common data loader module, using CRUD
# can be applied in every schemas and table schema
# +++++++++++++++++++++++++++++++++++++++++++++++++

T = TypeVar("T", bound="AbstractDatas")


class AbstractDatas(ABC, BaseModel):
    data: list

    @classmethod
    @abstractmethod
    def from_db(cls) -> T:
        """Should be placed with CRUD functions, reading DB data into BaseModel data"""
        pass

    def split_data(
        self: T,
    ) -> tuple[list, list]:
        """Split the data into testing data and training data."""
        # Divide the data into train and test groups
        train_data = []
        test_data = []

        for item in self.data:
            if item.label == "train":
                train_data.append(item)
            else:
                test_data.append(item)
        return train_data, test_data
