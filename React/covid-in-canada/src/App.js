import { useState, useEffect } from 'react'
import { Container, Typography, Icon, Grid, BottomNavigation } from '@material-ui/core';
import { Line } from 'react-chartjs-2'

import CountryPicker from './components/RegionPicker/RegionPicker'
import Cards from './components/Cards/Cards'
import covidImage from './covid-19.svg'
import loadingGif from './loading.gif'

// the number of previous day (minus 7) to use when comptuing 7 day case averages. 
const day_window = 35
// Add function to date object for generating graph labels
Date.prototype.addDays = function (days) {
  var date = new Date(this.valueOf());
  date.setDate(date.getDate() + days);
  return date;
}
// generate date labels for graphs
const dateArray = new Array()
var currentDate = new Date(new Date() - ((day_window - 8) * 24 * 60 * 60 * 1000))
while (currentDate <= new Date()) {
  dateArray.push(new Date(currentDate).toLocaleDateString());
  currentDate = currentDate.addDays(1);
}

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

  const data = {
    labels: dateArray,
    datasets: [
      {
        label: 'Daily Cases',
        data: timeSeriesNewCases,
        fill: false,
        backgroundColor: 'rgb(255, 99, 132)',
        borderColor: 'rgba(255, 99, 110, 0.9)',
      },
      {
        label: 'Active Case Count',
        data: timeSeriesActiveCases,
        fill: false,
        backgroundColor: 'rgb(10, 110, 255)',
        borderColor: 'rgba(10, 132, 255, 0.9)',
      },
    ],
  };

  const options = {
    // responsive: true,
    scales: {
      xAxes: [{
        ticks: {
          autoSkip: false,
          maxRotation: 90,
          minRotation: 90
        },
        scaleLabel: {
          display: true,
          labelString: 'Days'
        }
      }],
      yAxes: [{
        display: true,
        scaleLabel: {
          display: true,
          labelString: '7-day average'
        }
      }]
    },
  };

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
    const start = new Date(new Date() - ((day_window - 1) * 24 * 60 * 60 * 1000))
    // form: DD-MM-YYYY
    const after = `${start.getDate()}-${start.getMonth() + 1}-${start.getFullYear()}`
    console.log('day start 7-day average at:', after)
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
      var sum = 0.0
      var i = 0
      var j = 0
      var temp = []
      for (i = 0; i < cases.length - 7; i++) {
        sum = 0.0
        for (j = i; j < i + 7; j++) {
          sum += cases[j].cases
        }
        temp[i] = sum / 7.0
      }
      console.log('daily cases:', temp)
      setTimeSeriesNewCases(temp)

      // get time series with active cases
      res = await fetch(`${url}timeseries?stat=active&loc=${region}&after=${after}`)
      const { active } = await res.json().then(res => {
        return res
      })

      temp = []
      for (i = 0; i < active.length - 7; i++) {
        sum = 0.0
        for (j = i; j < i + 7; j++) {
          sum += active[j].active_cases
        }
        temp[i] = sum / 7.0
      }
      console.log('active cases:', temp)
      setTimeSeriesActiveCases(temp)

      console.log('days:', dateArray)
      setLoading(false)
    }

    fetchData()

  }, [region])


  return (
    <div>
      <Grid container direction='row' justify='center' alignItems='flex-end'>
        <Grid item style={{ marginRight: '20px', marginTop: '100px', marginBottom: '20px' }}>
          <img alt='covid' src={covidImage} style={{ width: 150, height: 150 }} />
        </Grid>
        <Grid item>
          <Typography color="textPrimary" variant='h2' align='center' gutterBottom>
            Covid In Canada
        </Typography>
        </Grid>
      </Grid>
      {
        loading ? (
          <Grid container item
            direction='row'
            justify='center'
            alignItems='center'
            style={{ margin: '40px 0px 40px 0px' }}>
            <img src={loadingGif} style={{ width: 200, height: 200 }} />
          </Grid>
        ) : (
          <div className="App">
            <CountryPicker
              currentRegion={region}
              handleSetRegion={handleSetRegion}
            />
            <Cards
              summaryData={summaryData}
              population={populations[region]}
            />
            <Typography color="textPrimary" variant="h4" align='center' gutterBottom>
              7-Day Averages over the Last Month
        </Typography>
            <Container minHeight="100%">
              <Line
                data={data}
                options={options} />
            </Container>
          </div>
        )
        // end main
      }
      <BottomNavigation />
    </div>
  )
}

export default App;
