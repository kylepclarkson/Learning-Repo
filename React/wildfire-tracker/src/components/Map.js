import useState from 'react'

import GoogleMapReact from 'google-map-react'

import LocationMarker from './LocationMarker'
import LocationInfoBox from './LocationInfoBox'

const Map = ({ eventData, center, zoom }) => {

    const [locationInfo, setLocationInfo] =  useState(null)

    // get wildfires
    const markers = eventData.map(ev => {
        if (ev.categories[0].id === 8) {
            return <LocationMarker 
                lat={ev.geometries[0].coordinates[1]} 
                lng={ev.geometries[0].coordinates[0]} 
                onClick={() => setLocationInfo(
                    {id: ev.id, title: ev.title}
                )}/>
        }
        return null
    })
    
    return (
        <div className="map">
            <GoogleMapReact
                bootstrapURLKeys={{ key: ''}}
                defaultCenter = {center} 
                defaultZoom= {zoom}
            >
                {markers}
            </GoogleMapReact>
            {/* if we have location info, display. */}
            {locationInfo && <LocationInfoBox info={locationInfo}/>}
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
