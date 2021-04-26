import React from 'react'
import { Card, CardContent, Grid, Typography } from '@material-ui/core';
import { withStyles } from '@material-ui/core/styles'
import CountUp from 'react-countup'
import cx from 'classnames'

import styles from './Cards.module.css'
import { red } from '@material-ui/core/colors';

const FontColors = withStyles({
  red: {
    color: "#FF0000"
  },
  green: {
    color: "#00FF00"
  }
})(Typography)

function Cards({ summaryData }) {
  return (
    <div>
      <div className={styles.container}>
        <Typography color="textPrimary" variant="h3" align='center' gutterBottom>
          Summary Data
        </Typography>
        <Grid container justify='center'>
          {/* Total Cases */}
          <Grid item component={Card} xs={12} sm={3} className={cx(styles.card, styles.cases)}>
            <CardContent>
              <Typography color="textSecondary" variant="h5" gutterBottom>
                Cases
            </Typography>
              <Typography variant='h5'>
                <CountUp
                  start={0}
                  end={summaryData.cumulative_cases}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography variant='body2'>
                Number of diagnosed cases of COVID-19.
            </Typography>
            </CardContent>
          </Grid>
          {/* Total Recovered */}
          <Grid item component={Card} xs={12} sm={3} className={cx(styles.card, styles.recovered)}>
            <CardContent>
              <Typography color="textSecondary" variant="h5" gutterBottom>
                Recovered
            </Typography>
              <Typography variant='h5'>
                <CountUp
                  start={0}
                  end={summaryData.cumulative_recovered}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography variant='body2'>
                Number of people who recovered from COVID-19.
            </Typography>
            </CardContent>
          </Grid>
          {/* Total Deaths */}
          <Grid item component={Card} xs={12} sm={3} className={cx(styles.card, styles.deaths)}>
            <CardContent>
              <Typography color="textSecondary" variant="h5" gutterBottom>
                Deaths
            </Typography>
              <Typography variant='h5'>
                <CountUp
                  start={0}
                  end={summaryData.cumulative_deaths}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography variant='body2'>
                Number of deaths caused by COVID-19.
            </Typography>
            </CardContent>
          </Grid>
        </Grid>
      </div>
      <div className={styles.container}>
        <Typography color="textPrimary" variant="h3" align='center' gutterTop gutterBottom>
          Within the last 24 hours
        </Typography>
        <Grid container justify='center'>
          {/* Total Cases */}
          <Grid item component={Card} xs={12} sm={3} className={cx(styles.card, styles.active)}>
            <CardContent>
              <Typography color="textSecondary" variant="h5" gutterBottom>
                Active
            </Typography>
              <Typography variant='h5'>
                <CountUp
                  start={0}
                  end={summaryData.active_cases}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography variant='body2'>
                Number currently active cases of COVID-19.
            </Typography>
            </CardContent>
          </Grid>
          {/* Total Recovered */}
          <Grid item component={Card} xs={12} sm={3} className={cx(styles.card, styles.new)}>
            <CardContent>
              <Typography color="textSecondary" variant="h5" gutterBottom>
                New Cases
            </Typography>
              <Typography variant='h5'>
                <CountUp
                  start={0}
                  end={summaryData.cases}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography color="textSecondary" variant="body1" gutterBottom>
                Change
            </Typography>
              <Typography variant='body1' className={ summaryData.active_cases_change < 0 ? styles.red : styles.green}>
                <CountUp
                  start={0}
                  end={summaryData.active_cases_change}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography variant='body2'>
                The number new cases of COVID-19 and change since previous update.
            </Typography>
            </CardContent>
          </Grid>
          {/* Total Deaths */}
          <Grid item component={Card} xs={12} sm={3} className={cx(styles.card, styles.deaths)}>
            <CardContent>
              <Typography color="textSecondary" variant="h5" gutterBottom>
                New Deaths
            </Typography>
              <Typography variant='h5'>
                <CountUp
                  start={0}
                  end={summaryData.deaths}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography variant='body2'>
                Number of new deaths caused by COVID-19.
            </Typography>
            </CardContent>
          </Grid>
        </Grid>
      </div>
      <div className={styles.container}>
        <Typography color="textPrimary" variant="h3" align='center' gutterBottom>
          Vaccine Distribution
        </Typography>
        <Grid container justify='center'>
          {/* Total Cases */}
          <Grid item component={Card} xs={12} sm={3} className={cx(styles.card, styles.cases)}>
            <CardContent>
              <Typography color="textSecondary" variant="h5" gutterBottom>
                Delivered
              </Typography>
              <Typography variant='h5'>
                <CountUp
                  start={0}
                  end={summaryData.cumulative_dvaccine}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography variant='body2'>
                The number of vaccines delivered.
            </Typography>
            </CardContent>
          </Grid>
          {/* Total Recovered */}
          <Grid item component={Card} xs={12} sm={3} className={cx(styles.card, styles.recovered)}>
            <CardContent>
              <Typography color="textSecondary" variant="h5" gutterBottom>
                Partially Vaccinated
            </Typography>
              <Typography variant='h5'>
                <CountUp
                  start={0}
                  end={summaryData.cumulative_avaccine}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography variant='body2'>
                The number of individuals given at least one vaccine shot.
            </Typography>
            </CardContent>
          </Grid>
          {/* Total Deaths */}
          <Grid item component={Card} xs={12} sm={3} className={cx(styles.card, styles.deaths)}>
            <CardContent>
              <Typography color="textSecondary" variant="h5" gutterBottom>
                Fully Vaccinated
            </Typography>
              <Typography variant='h5'>
                <CountUp
                  start={0}
                  end={summaryData.cumulative_cvaccine}
                  duration={2}
                  separator=','
                />
              </Typography>
              <Typography variant='body2'>
                The number of individuals fully vaccinated.
            </Typography>
            </CardContent>
          </Grid>
        </Grid>
      </div>
    </div>
  )
}

export default Cards
