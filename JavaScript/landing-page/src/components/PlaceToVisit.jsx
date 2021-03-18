import React from 'react'
import { makeStyles } from '@material-ui/core/styles'
import ImageCard from './ImageCard'
import useWindowPosition from '../hook/useWindowPosition'

import places from '../static/places'

const useStyles = makeStyles((theme) => ({
    root: {
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        [theme.breakpoints.down('md')]: {
            flexDirection: 'column'
        }
    },
}))

function PlaceToVisit() {
    const classes = useStyles()
    const checked = useWindowPosition('header')

    return (
        <div className={classes.root} id='place-to-visit'>
            <ImageCard place={places[0]} checked={checked}/>
            <ImageCard place={places[1]} checked={checked}/>
        </div>
    )
}

export default PlaceToVisit
