import React from 'react'
import { Card, CardContent, Grid, Typography } from '@material-ui/core';
import CountUp from 'react-countup'
import cx from 'classnames'

import styles from './Cards.module.css'

function Cards(data) {

  if (data === null) {
    console.log("Data is null")
  } else {
    console.log(data)
  }

  return (
    <div className={styles.container}>
      <Grid container justify='center'>
        {/* Total Cases */}
        <Grid item component={Card} xs={12} sm={6} md={2} className={cx(styles.card, styles.cases)}>
          <CardContent>
            <Typography color="textSecondary" variant="h5" gutterBottom>
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
            <Typography variant='body2'>
              Number of diagnosed cases of COVID-19
            </Typography>
          </CardContent>
        </Grid>
        {/* Active Cases */}
        <Grid item component={Card} xs={12} sm={6} md={2} className={cx(styles.card, styles.active)}>
          <CardContent>
            <Typography color="textSecondary" variant="h5" gutterBottom>
              Active
            </Typography>
            <Typography variant='h5'>
              <CountUp
                start={0}
                end={100000}
                duration={3}
                separator=','
              />
            </Typography>
            <Typography variant='body2'>
              Number of active COVID-19 cases.
            </Typography>
          </CardContent>
        </Grid>
        {/* Total Recovered */}
        <Grid item component={Card} xs={12} sm={6} md={2} className={cx(styles.card, styles.recovered)}>
          <CardContent>
            <Typography color="textSecondary" variant="h5" gutterBottom>
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
            <Typography variant='body2'>
              Number of people recovered from COVID-19
            </Typography>
          </CardContent>
        </Grid>
        {/* Total Deaths */}
        <Grid item component={Card} xs={12} sm={6} md={2} className={cx(styles.card, styles.deaths)}>
          <CardContent>
            <Typography color="textSecondary" variant="h5" gutterBottom>
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
            <Typography variant='body2'>
              Number of deaths caused by COVID-19
            </Typography>
          </CardContent>
        </Grid>
      </Grid>
    </div>
  )
}

export default Cards
