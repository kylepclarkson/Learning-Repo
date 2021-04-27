import { useState, useEffect } from 'react'

import CountryPicker from './components/RegionPicker/RegionPicker'
import Cards from './components/Cards/Cards'
import { summary } from './api'

function App() {

  // summary data
  const [summaryData, setSummaryData] = useState([])
  // timeseries data
  const [timeSeriesData, setTimeSeriesData] = useState([])
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
    const date = new Date();
    // Get 28th day before current
    // const start = new Date(today.getTime() - (28*24*60*60*1000))
    date.setDate(date.getDate() - 28)
    console.log('start', date)
    // form: DD-MM-YYYY
    var after;
    // account for leap year.
    if (date.getDate() == 29 && date.getMonth() == 2) {
      after = `28-${date.getMonth()-1}-${date.getFullYear()}`
    }  else {
      after = `${date.getDate()}-${date.getMonth()-1}-${date.getFullYear()}`
    }
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

      console.log('after', after)
      res = await fetch(`${url}timeseries?stat=cases&loc=${region}&after=${after}`)
      const { cases } = await res.json().then(res => {
        return res
      })
      setTimeSeriesData(cases)
      console.log('timeseries: ', cases)
      console.log(cases.length)
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
