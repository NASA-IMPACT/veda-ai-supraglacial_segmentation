const mapConfig = {
  container: "map",
  style: "mapbox://styles/mapbox/light-v10",
  center: [-73.7732248, 67.8733544],
  zoom: 2,
}

const mapboxAccessToken = "pk.eyJ1Ijoic2xlc2FhZCIsImEiOiJjazIycnZvd3Iwb3VlM25tbDlnODB5azg1In0.oUpVrwzxHz_stwQFlUUCBQ";

const dataConfig = {
  name: "coverage",
  path: "../../data/",
  type: "geojson"
}

const vizConfig = {
  type: "fill",
  paint: {
    "fill-color": "#d35400",
    "fill-opacity": 0.1
  }
}

const locations = {
  "Negribreen, Svalbard": [19.1324906, 78.5638881],
  "Jakobshavns Isbrae": [-49.9341761, 69.1666655],
  "Sverdrup Glacier": [-83.1841761, 75.6666658],
  "Giesecke Braer": [-55.2500000, 73.58333],
  "Helheim Glacier": [-38.2175095, 66.3499986],
}
