import {useState, useEffect } from 'react'

import './App.css';

function App() {

  const API_BASE = 'https://api.opencovid.ca/'

  // data retrieved from API
  const [covidData, setCovidData] = useState([])
  // true when data is being retrieved.
  const [loading, setLoading] = useState(false)

  /* 
    useEffect(func) A function that is ran after every render commmitted to the screen.
      Would take functionality found in componentDidMount, componentDidUpdate, and componentWillUnmount.
  */

  useEffect(() => {
    
    // get covid data.
    const fetchData = async() => {
        setLoading(true)

        const res = await fetch(API_BASE)
        const data = await res.json()

        setCovidData(data)
        setLoading(false)
        console.log("Data", data)
    }

    fetchData()
    console.log(`${API_BASE}/summary`)
  }, [])

  return (
    <div className="App">
      { !loading ? <h1>Data</h1> : <h1>Loading</h1>}
      {console.log(covidData)}
    </div>
  );
}

export default App;
