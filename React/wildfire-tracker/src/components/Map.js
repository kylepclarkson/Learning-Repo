import { useState } from 'react'
import GoogleMapReact from 'google-map-react'

const Map = ({ eventData, center, zoom}) => {

    // TODO get datafrom api. Create location marks, and create locationInfo state. 

    return (
        <div className="map">
            <GoogleMapReact
                bootstrapURLKeys={{ key: 'AIzaSyBJTS1eMEBBO2_RX8q7gbScBS_jaqEk1BI'}}
                defaultCenter={center}
                defaultZoom={zoom}
            >

            </GoogleMapReact>
        </div>
    )
}

// Map default center and zoom 
Map.defaultProps = {
    center: {
        lat: 48.3,
        lng: -122,
    },
    zoom: 5,
}

export default Map
