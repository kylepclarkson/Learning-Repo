
import React, { Component } from 'react'
import styles from './App.module.css'

import { Cards, Chart, CountryPicker } from './components'
import { fetchData } from './api'

export class App extends Component {

    state = {
        data: {},
    }

    async componentDidMount() {
        // get data from api
        const data = await fetchData()
        this.setState({ data: data})
        console.log(data)
    }
    render() {

        const { data } = this.state

        return (
            <div className={styles.container}>
                <Cards data={data}/>
                <CountryPicker />
                <Chart />
            </div>
        )
    }
}

export default App
