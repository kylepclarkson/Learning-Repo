
import {useState, useEffect} from 'react'

function useStats() {
    const[stats, useStats] = useState()

    useEffect(() => {
        console.log('fetch data')
    })
}

