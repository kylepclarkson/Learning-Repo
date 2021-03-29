// handles fetching of data from API

import axios from 'axios'

const url = 'https://api.opencovid.ca/'

export const fetchData = async () => {

    try {
        const { data } = await axios.get(url);
        
        const send = {
            summary: data.summary,
        }
        return send;

    } catch (error) {
        console.log(error);
    }
}
