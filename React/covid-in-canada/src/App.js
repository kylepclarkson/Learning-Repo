import { useState, useEffect } from 'react'

import CountryPicker from './components/RegionPicker/RegionPicker'
import Cards from './components/Cards/Cards'
import { summary } from './api'

function App() {

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
    const fetchData = async () => {
      setLoading(true)
      const res = await summary(region)
      const { summary } = await res.json().then(res => {
        return res
      })
      console.log("summary", summary)
      setCovidData(summary)
      setLoading(false)
    }

    fetchData()
    console.log('Region: ', region)
    console.log('Summary data: ', covidData)
  }, [])

  if (loading) {
    return <h1>Loading</h1>
  } else {
    console.log("Finally: ", covidData)
    return (
      <div className="App">
        <h1>{region}</h1>
        <CountryPicker onRegionSelect={region => setRegion(region)} />
        <Cards
          data={covidData}
        />
      </div>
    );
  }
}

export default App;
