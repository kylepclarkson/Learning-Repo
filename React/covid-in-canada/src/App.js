import { useState, useEffect } from 'react'

import CountryPicker from './components/RegionPicker/RegionPicker'
import Cards from './components/Cards/Cards'
import { summary } from './api'

function App() {

  // summary data
  const [summaryData, setSummaryData] = useState([])
  // timeseries new cases data
  const [timeSeriesNewCases, setTimeSeriesNewCases] = useState([])
  // timeseries active cases data
  const [timeSeriesActiveCases, setTimeSeriesActiveCases] = useState([])
  // true when data is being retrieved.
  const [loading, setLoading] = useState(false)
  // region of country
  const [region, setRegion] = useState('canada')
  // last 21 days of seven day case averages. 

  // population estimates for 2021Q1 from gov.
  // https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000901
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
    const url = 'https://api.opencovid.ca/'
    // Get 28th day before current
    const start = new Date(new Date() - (27*24*60*60*1000))
    // form: DD-MM-YYYY
    const after = `${start.getDate()}-${start.getMonth()+1}-${start.getFullYear()}`
    
    // Call API for data.
    const fetchData = async () => {
      setLoading(true)
      // Get summary data for region.
      var res = await fetch(`${url}summary?loc=${region}`)
      const { summary } = await res.json().then(res => {
        return res
      })
      console.log("summary", summary[0])
      setSummaryData(summary[0])

      // get time series with daily new cases
      res = await fetch(`${url}timeseries?stat=cases&loc=${region}&after=${after}`)
      const { cases } = await res.json().then(res => {
        return res
      })
      setTimeSeriesNewCases(cases)
      console.log('new cases: ', cases)
      
      // get time series with active cases
      res = await fetch(`${url}timeseries?stat=active&loc=${region}&after=${after}`)
      const { active } = await res.json().then(res => {
        return res
      })
      setTimeSeriesActiveCases(active)
      console.log('active cases: ', active)

      setLoading(false)
    }

    fetchData()

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
