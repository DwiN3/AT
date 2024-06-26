from backend.CovidCase import CovidCase


class CovidData:
    def __init__(self, country: str, region: str, cases: [CovidCase]) -> None:
        self.country = country
        self.region = region
        self.cases = cases