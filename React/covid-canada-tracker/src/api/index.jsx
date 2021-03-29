// handles fetching of data from API

import axios from 'axios'

const url = 'https://api.opencovid.ca/'

export const fetchData = async () => {

    try {
        const { data } = await axios.get(url);
        return data

    } catch (error) {
        console.log(error);
    }
}
