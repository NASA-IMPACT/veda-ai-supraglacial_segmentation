const monthsText = ["May", "June", "July", "Aug", "Sept", "Oct"];
const months = [5, 6, 7, 8, 9, 10];
const years = [2016, 2017, 2018, 2019, 2020, 2021];

const mapping = (index) => {
  let year = years[parseInt(index / months.length)];
  let monthIndex = index % months.length;
  return {
    text: `${monthsText[monthIndex]} ${year}`,
    year: year,
    month: months[monthIndex],
  };
};

document.addEventListener("DOMContentLoaded", function () {
  // MapboxGL stuff

  mapboxgl.accessToken = mapboxAccessToken;
  const map = new mapboxgl.Map(mapConfig);
  const markers = {};

  function filterBy(index, location) {
    let time = mapping(index);
    let year = time.year,
      month = time.month;
    const filters = ["all", ["==", "year", year], ["==", "month", month]];
    map.setFilter(`${dataConfig.name}-${location}-layer`, filters);

    // Set the label to the time
    document.getElementById("time").textContent = mapping(index).text;
  }
  // Reading the data
  map.on("load", () => {
    for (const location in locations) {
      markers[location] = new mapboxgl.Marker()
                                    .setLngLat(locations[location])
                                    .addTo(map);
      markers[location].getElement().addEventListener('click', () => {
        map.flyTo({
          center: locations[location],
          zoom: 7,
          speed: 0.8
        })

        // clean up
        if (!(map.getLayer(`${dataConfig.name}-${location}-layer`)) || (!map.getSource())) {
          // map.removeLayer(`${dataConfig.name}-${location}-layer`);
          //load new data
          d3.json(`${dataConfig.path}${location}.geojson`, (err, data) => {
            jsonCallback(err, data, location)
          });
        }
      })
    }
    
  });

  function jsonCallback(err, data, location) {
    if (err) {
      throw err;
    }

    map.addSource(`${dataConfig.name}-${location}`, {
      type: dataConfig.type,
      data: data,
    });

    map.addLayer({
      id: `${dataConfig.name}-${location}-layer`,
      source: `${dataConfig.name}-${location}`,
      ...vizConfig
    });

    filterBy(0, location);

    document.getElementById("slider").addEventListener("input", (e) => {
      const time = parseInt(e.target.value, 10);
      filterBy(time, location);
    });
  }
});
