import React from 'react'
import { Card, CardContent, Grid, Typography } from '@material-ui/core';
import CountUp from 'react-countup'
import cx from 'classnames'

import styles from './Cards.module.css'

function Cards() {

  return (
    <div className={styles.container}>
      <Grid container spacing={3} justify='center'>
        {/* Total Cases */}
        <Grid item component={Card} xs={12} md={3} className={cx(styles.card, styles.cases)}>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Cases
            </Typography>
            <Typography variant='h5'>
              <CountUp
                start={0}
                end={100000}
                duration={3}
                separator=','
              />
            </Typography>
            <Typography color="textSecondary" gutterBottom>
              DATE
            {/* {new Date(lastUpdate).toDateString()} */}
            </Typography>
            <Typography variant='body2'>
              Number of diagnosed cases of COVID-19
            </Typography>
          </CardContent>
        </Grid>
        {/* Total Deaths */}
        <Grid item component={Card} xs={12} md={3} className={cx(styles.card, styles.deaths)}>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Deaths
            </Typography>
            <Typography variant='h5'>
              <CountUp
                start={0}
                end={100000}
                duration={3}
                separator=','
              />
            </Typography>
            <Typography color="textSecondary" gutterBottom>
              DATE
            {/* {new Date(lastUpdate).toDateString()} */}
            </Typography>
            <Typography variant='body2'>
              Number of deaths caused by COVID-19
            </Typography>
          </CardContent>
        </Grid>
        {/* Total Recovered */}
        <Grid item component={Card} xs={12} md={3} className={cx(styles.card, styles.recovered)}>
          <CardContent>
            <Typography color="textSecondary" gutterBottom>
              Recovered
            </Typography>
            <Typography variant='h5'>
              <CountUp
                start={0}
                end={100000}
                duration={3}
                separator=','
              />
            </Typography>
            <Typography color="textSecondary" gutterBottom>
              DATE
            {/* {new Date(lastUpdate).toDateString()} */}
            </Typography>
            <Typography variant='body2'>
              Number of people recovered from COVID-19
            </Typography>
          </CardContent>
        </Grid>
      </Grid>
    </div>
  )
}

export default Cards
