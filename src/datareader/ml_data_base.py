from sklearn.model_selection import train_test_split
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
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> tuple[list, list]:
        """Split the data into testing data and training data."""
        train_data, test_data = train_test_split(
            self.data, test_size=test_size, random_state=random_state
        )
        return train_data, test_data
