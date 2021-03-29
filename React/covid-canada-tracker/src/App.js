import React from 'react'

import {Cards, Charts, ProvincePicker} from './components'
import styles from './App.module.css';
import {fetchData} from './api/index'

class App extends React.Component {

  state = {
    data: {},

  }

  async componentDidMount() {
    // get api data
    const fetchedData = await fetchData();
    this.setState({
      data: fetchedData
    })
  }

  render() {

    const { data } = this.state

    return (
      <div className={styles.container}>
        <Cards data={data}/>
        <ProvincePicker />
        <Charts />
      </div>
    )
  }
}

export default App;