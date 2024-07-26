from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import TypeVar
from datetime import datetime
from enum import Enum
# +++++++++++++++++++++++++++++++++++++++++++++++++
# common data loader module, using CRUD
# can be applied in every schemas and table schema
# +++++++++++++++++++++++++++++++++++++++++++++++++

T = TypeVar("T", bound="AbstractDatas")


class Label(str, Enum):
    train = "train"
    test = "test"


class BaseData(BaseModel):
    user_id: int
    movie_id: int
    rating: int
    movie_title: str
    timestamp: datetime
    label: Label


class AbstractDatas(ABC, BaseModel):
    data: list[BaseData]

    @classmethod
    @abstractmethod
    def from_db(cls) -> T:
        """Should be placed with CRUD functions, reading DB data into BaseModel data"""
        pass

    @classmethod
    def from_input(cls, input_data: list[BaseData]):
        return cls(data=input_data)

    def split_data(
        self: T,
    ) -> tuple[list[BaseData], list[BaseData]]:
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
