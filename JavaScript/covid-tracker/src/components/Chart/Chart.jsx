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

    // line chart for global numbers
    const lineChart = (
        // if dailyData exists create line component else null. 
        dailyData.length !== 0 ? (
            <Line data={{
                labels: dailyData(({ date }) => date),
                datasets: [
                    {
                        data: dailyData(({ confirm }) => confirm),
                        label: 'Infected',
                        borderColor: '#3333ff',
                        fill: true
                    },
                    {
                        data: dailyData(({ deaths }) => deaths),
                        label: 'Infected',
                        borderColor: 'red',
                        backgroundColor: 'rgba(255, 0, 0, 0.5)',
                        fill: true
                    }
                ]
        }}/> ) : null
    )


    return (
        <div className="styles.container">
            <lineChart />
        </div>
    )
}

export default Chart
