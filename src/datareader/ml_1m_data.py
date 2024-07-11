import asyncio
from datetime import datetime
from pathlib import Path

from datareader.db import fetch_ratings
from pydantic import BaseModel


class Rating(BaseModel):
    user_id: int
    item_id: int
    rating: int
    timestamp: datetime


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

    @classmethod
    async def from_db(cls) -> "Ratings":
        ratings = await fetch_ratings()
        read_data = [
            Rating(
                user_id=rating.user_id,
                item_id=rating.item_id,
                rating=rating.rating,
                timestamp=rating.timestamp,
            )
            for rating in ratings
        ]
        return cls(data=read_data)


if __name__ == "__main__":
    # print(Ratings.from_csv().data[0:10])
    async def _main():
        # caution: this will produce tons of output
        ratings = await Ratings.from_db()
        print(ratings.data)

    asyncio.run(_main())
