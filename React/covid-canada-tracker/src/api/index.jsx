// handles fetching of data from API

import axios from 'axios'

const url = 'https://api.opencovid.ca/'

/**
 * Get summary (test) data.
 * 
 */
export const fetchData = async () => {

    try {
        const { data } = await axios.get(url);
        return data

    } catch (error) {
        console.log(error);
    }
}

// Get data for today
export const fetchDailyData = async() => {
    try {
        const response = await axios.get(`${url}/`)
    } catch (error) {
        console.log(error)
    }
}

