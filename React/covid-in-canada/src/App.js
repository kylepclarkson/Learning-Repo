import { useState, useEffect } from 'react'

import CountryPicker from './components/RegionPicker/RegionPicker'
import Cards from './components/Cards/Cards'
import { summary } from './api'

function App() {

  // data retrieved from API
  const [summaryData, setSummaryData] = useState([])
  // true when data is being retrieved.
  const [loading, setLoading] = useState(false)
  // region of country
  const [region, setRegion] = useState('canada')

  const populations = {
    'canada': 38048738,
    'AB': 4436258,
    'BC': 5153039,
    'MB': 1380935,
    'NB': 1178832,
    'NL': 520438,
    'NT': 45136,
    'NS': 979449,
    'NU': 39407,
    'ON': 14755211,
    'PE': 159819,
    'QC': 8575944,
    'SK': 1178832,
    'YT': 42192,
  }

  /* 
    useEffect(func) A function that is ran after every render commmitted to the screen.
      Would take functionality found in componentDidMount, componentDidUpdate, and componentWillUnmount.
  */

  const handleSetRegion = r => {
    console.log("Setting region", r)
    setRegion(r)
    console.log(region)
  } 

  useEffect(() => {
    console.log('use effect called')
    const url = 'https://api.opencovid.ca/'
    
    // Call API for data.
    const fetchData = async () => {
      setLoading(true)
      // Get summary data for region.
      const res = await fetch(`${url}summary?loc=${region}`)
      const { summary } = await res.json().then(res => {
        return res
      })
      console.log("summary", summary[0])
      setSummaryData(summary[0])
      setLoading(false)
    }

    fetchData()
    console.log('Region: ', region)

    const today = new Date();
    const start = new Date(today.getTime() - (28*24*60*60*1000))
    console.log(start.getDate(), start.getMonth()+1, start.getFullYear())

  }, [region])

  if (loading) {
    return <h1>Loading</h1>
  } else {
    return (
      <div className="App">
        <CountryPicker 
          currentRegion = {region}
          handleSetRegion={handleSetRegion}
        />
        <Cards
          summaryData={summaryData}
          population={populations[region]}
        />
      </div>
    );
  }
}

export default App;
