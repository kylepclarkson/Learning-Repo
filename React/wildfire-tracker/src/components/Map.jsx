import GoogleMapReact from 'google-map-react'

import LocationMarker from './LocationMarker'

const Map = ({ center, zoom }) => {

    
    return (
        <div className="map">
            <GoogleMapReact
                bootstrapURLKeys={{ key: 'AIzaSyBJTS1eMEBBO2_RX8q7gbScBS_jaqEk1BI'}}
                defaultCenter = {center}
                defaultZoom= {zoom}
            >
                <LocationMarker 
                    lat={center.lat}
                    lng={center.lng} />
            </GoogleMapReact>
        </div>
    )
}

// The default location and zoom, when page is loaded. 
Map.defaultProps = {
    center: {
        lat: 48.3245,
        lng: -122.8765
    },
    zoom: 6,
}

export default Map
