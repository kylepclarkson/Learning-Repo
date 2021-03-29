import React from 'react'

import { Card, CardContent, Typography, Grid} from '@material-ui/core';
import CountUp from 'react-countup'
import cx from 'classnames';

import styles from './Cards.module.css'

const Cards = (props) => {
    if (!props.data.summary) {
        // data not yet loaded. 
        return <h1>Loading</h1>
    }
    console.log(props.data.summary[0])

    const {cumulative_cases, cumulative_deaths, cumulative_recovered } = props.data.summary[0]
    console.log(cumulative_cases)
    return (
        <div className={styles.container}>
            <Grid container spacing={3} justify="center">
                <Grid item component={Card} xs={12} md={3} className={cx(styles.card, styles.infected)}>
                    <CardContent>
                        <Typography color="textSecondary" gutterBottom>Infected</Typography>
                        <Typography variant="h5">
                            <CountUp
                                start={0}
                                end={cumulative_cases}
                                duration={2.5}
                                separator=",">
                            </CountUp>
                            </Typography>
                        <Typography variant="body2">Active number of active</Typography>
                    </CardContent>
                </Grid>
                <Grid item component={Card} xs={12} md={3} className={cx(styles.card, styles.recovered)}>
                    <CardContent>
                        <Typography color="textSecondary" gutterBottom>Recovered</Typography>
                        <Typography variant="h5">
                            <CountUp
                                start={0}
                                end={cumulative_recovered}
                                duration={2.5}
                                separator=",">
                            </CountUp>
                            </Typography>
                        <Typography variant="body2">Active number of recoveries</Typography>
                    </CardContent>
                </Grid>
                <Grid item component={Card} xs={12} md={3} className={cx(styles.card, styles.deaths)}>
                    <CardContent>
                        <Typography color="textSecondary" gutterBottom>Deaths</Typography>
                        <Typography variant="h5">
                            <CountUp
                                start={0}
                                end={cumulative_deaths}
                                duration={2.5}
                                separator=",">
                            </CountUp>
                            </Typography>
                        <Typography variant="body2">Active number of deaths</Typography>
                    </CardContent>
                </Grid>
            </Grid>
        </div>
    )
}

export default Cards;