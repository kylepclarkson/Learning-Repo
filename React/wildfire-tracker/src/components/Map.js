import { useState } from 'react'
import GoogleMapReact from 'google-map-react'
import LocationMarker from './LocationMarker'
import LocationInfoBox from './LocationInfoBox'

const Map = ({ eventData, center, zoom}) => {

    const [locationInfo, setLocationInfo] = useState(null)

    // get locations of fires
    const markers = eventData.map((ev, index) => {
        // id of 8 designates event is of type wildfire. 
        if (ev.categories[0].id === 8) {
            // Create location markers with location info
            return (
                <LocationMarker
                    key={index}
                    lat={ev.geometries[0].coordinates[1]}
                    lng={ev.geometries[0].coordinates[0]}
                    onClick={() => setLocationInfo({id: ev.id, title: ev.title})}
                />
            )
        }
        return null
    })

    return (
        <div className="map">
            <GoogleMapReact
                bootstrapURLKeys={{ key: 'AIzaSyBJTS1eMEBBO2_RX8q7gbScBS_jaqEk1BI'}}
                defaultCenter={center}
                defaultZoom={zoom}
            >
            {/* display markers */}
                {markers}
            </GoogleMapReact>
            {locationInfo && <LocationInfoBox info={locationInfo} />}
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
