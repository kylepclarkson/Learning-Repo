import React from 'react'

import {Cards, Charts, ProvincePicker} from './components'
import styles from './App.module.css';
import {fetchData} from './api/index'
import {useState, useEffect} from 'react'

function useStats() {
    const[stats, useStats] = useState()

    useEffect( async() => {
        console.log('fetch data')
        fetchData('https://covid19.mathdro.id/api').then(
          data => data.json()
        );
    })
}


class App extends React.Component {
  async componentDidMount() {
  }

  render() {

    return (
      <div>
        <p>Hello</p>
      </div>
    )
  }
}

export default App;