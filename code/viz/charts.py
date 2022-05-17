from collections import defaultdict
import json

AOIs = {
    "Negribreen, Svalbard": [18.747975,78.549513,19.896046,78.677482],
    "Jakobshavns Isbrae": [-50.446716,69.070741,-49.453827,69.303492],
    "Sverdrup Glacier": [-86.454699,74.407512,-79.313586,76.273417],
    "Giesecke Braer": [-55.577168,73.482949,-55.088276,73.624472],
    "Helheim Glacier": [-38.427557,66.247253,-37.615942,66.520674],
    "Bering glacier system": [-143.712158203125, 59.91097597079679, -142.2894287109375, 60.557278971727264],
    "Ellesmere Island": [-77.798223,82.969529,-73.271856,83.261832]
}

stats = defaultdict(dict)

for k in AOIs:
    geojson = json.load(open(f'data/{k}.geojson'))
    features = geojson['features']

    for feature in features:
        properties = feature["properties"]
        month, year = properties["month"], properties["year"]

        if year in stats[k]:
            if month in stats[k][year]:
                stats[k][year][month] += 1
            else:
                stats[k][year][month] = 0
        else:
            stats[k][year] = {}
            stats[k][year][month] = 0

json.dump(stats, open('data/stats.json', 'w'))

monthsText = ["May", "June", "July", "Aug", "Sept", "Oct"];
months = [5, 6, 7, 8, 9, 10];
years = [2016, 2017, 2018, 2019, 2020, 2021];

# Creating data structure accepted by high charts
series = []
for k in AOIs:
    seri = {}
    seri["name"] = k
    stat = []
    for year in years:
        for month in months:
            stat.append(stats[k].get(year, {}).get(month, 0))
    seri["data"] = stat
    series.append(seri)

print(json.dumps(series, indent=4))
