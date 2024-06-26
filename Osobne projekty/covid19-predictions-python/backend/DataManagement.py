import json
import requests


def load_data(country: str) -> list:
    '''Loads COVID-19 data for a given country from the API'''
    api_url = f'https://api.api-ninjas.com/v1/covid19?country={country}'
    response = requests.get(api_url, headers={'X-Api-Key': 'ysJcUE/TeWo6yRbktNfNtw==1iEmOIxBoXTUg5PP'})

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code} {response.reason}")

    data = response.json()
    return data


def load_countries() -> list:
    '''Loads the list of countries from a JSON file'''
    with open('../frontend/data/Countries.json', 'r') as file:
        data = json.load(file)
    return data['countries']
