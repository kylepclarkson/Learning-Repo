import React, {useState, useEffect} from 'react'
import { fetchDailyData } from '../../api'
import {Line, Bar } from 'react-chartjs-2' 

const Chart = () => {

    const [dailyData, setDailyData] = useState([]);

    useEffect(() => {
        // create function to run async.
        const fetchAPI = async () => {
            const dailyData = await fetchDailyData();
            setDailyData(dailyData)
        }

        fetchAPI();
    });

    return (
        <div className="styles.container">
            <lineChart />
        </div>
    )
}

export default Chart
