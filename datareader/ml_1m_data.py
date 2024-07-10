from pydantic import BaseModel
from pathlib import Path


class Rating(BaseModel):
    user_id: int
    item_id: int
    rating: int
    timestamp: int


class Ratings(BaseModel):
    data: list[Rating]

    @classmethod
    def from_csv(cls, filepath: Path | str) -> "Ratings":
        # Load the ratings data
        # filepath is supposed to be data_dir / "ratings.dat"
        with open(filepath) as f:
            rows = f.readlines()

        read_data = []
        for row in rows:
            row_clean = row.replace("\n", "")
            split_row = row_clean.split("::")
            read_data.append(
                Rating(
                    user_id=int(split_row[0]),
                    item_id=int(split_row[1]),
                    rating=int(split_row[2]),
                    timestamp=int(split_row[3]),
                )
            )
        return cls(data=read_data)


if __name__ == "__main__":
    print(Ratings.from_csv().data[0:10])
