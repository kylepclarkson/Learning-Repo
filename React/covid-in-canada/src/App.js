import {useState, useEffect } from 'react'

import CountryPicker from './components/RegionPicker/RegionPicker'
import {summary} from './api'

import './App.css';



function App() {

  const API_BASE = 'https://api.opencovid.ca/'

  // data retrieved from API
  const [covidData, setCovidData] = useState([])
  // true when data is being retrieved.
  const [loading, setLoading] = useState(false)
  // region of country
  const [region, setRegion] = useState('canada')
  /* 
    useEffect(func) A function that is ran after every render commmitted to the screen.
      Would take functionality found in componentDidMount, componentDidUpdate, and componentWillUnmount.
  */

  useEffect(() => {
    console.log('use effect called')
    // get covid data.
    const fetchData = async() => {
        setLoading(true)
        setCovidData(summary(region))
    }
    
    fetchData()
    console.log('Region: ', region)
    console.log('Data: ', covidData)
  }, [])

  return (
    <div className="App">
      <h1>{region}</h1>
      <CountryPicker onRegionSelect={region => setRegion(region)}/>
    </div>
  );
}

export default App;
