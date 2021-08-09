mapboxgl.accessToken = 'pk.eyJ1IjoiZHdpbmlzc2FhIiwiYSI6ImNraWN3dHNkdDB3N20yeXBtNTJlbjk1bzIifQ.bD7wKzRtQPVWVnSTsjNXcA';
var map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/streets-v9',
    //'mapbox://styles/mapbox/dark-v10',
    center: [112.8, -7.6],
    zoom: 7
});
fetch('DATA/GeoObs_FINAL.json').then(response => response.json()).then(data => {
    map.on('load', function() {
        map.addSource('points', {
            'type': 'geojson',
            'data': data,
        });
        // Add a symbol layer
        map.addLayer({
            'id': 'points-yeah',
            'type': 'symbol',
            'source': 'points',
        });

        data.features.forEach(GeolocationCoordinates => {
                const marker = new mapboxgl.Marker({})
                    .setLngLat([GeolocationCoordinates.geometry.coordinates[1], GeolocationCoordinates.geometry.coordinates[0]])
                    .setPopup(new mapboxgl.Popup({
                            offset: 25
                        })
                        .setHTML("<p> <span style='font-weight:bold'>Loc &emsp;&emsp;</span>: " + GeolocationCoordinates.properties.location + "\n </p>" +
                            "<p> <span style='font-weight:bold'>Points &emsp;</span>: (" + GeolocationCoordinates.geometry.coordinates[0] + "," + GeolocationCoordinates.geometry.coordinates[1] + ")<p>"))
                    .addTo(map)
            })
            // Add zoom and rotation controls to the map.
        map.addControl(new mapboxgl.NavigationControl());
    });
})