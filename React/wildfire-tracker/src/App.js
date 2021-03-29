import {useState, useEffect } from 'react'

import Map from './components/Map'

function App() {

  const [eventData, setEventData ] = useState([]);
  const [loading, setLoading] = useState([]);

  useEffect(() => {
    const fetchEvents = async() => {
      setLoading(true);
      const resp = await fetch('https://eonet.sci.gsfc.nasa.gov/api/v2.1/events')
      // array of events
      const { events } = await resp.json()

      setEventData(events);
      setLoading(false);
      console.log(eventData);
    }

    
    fetchEvents();

  }, [])


  return (
    <div>
      { !loading ? <Map /> : <h1>Loading</h1>}
    
    </div>
  );
}

export default App;
