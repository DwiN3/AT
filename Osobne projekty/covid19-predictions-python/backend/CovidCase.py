
class CovidCase:
    def __init__(self, date: str, total: int, new: int) -> None:
        self.date = date
        self.total = total
        self.new = new

    def __repr__(self):
        return f"CovidCase(date={self.date}, total={self.total}, new={self.new})"