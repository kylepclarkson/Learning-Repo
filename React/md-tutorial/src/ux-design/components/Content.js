import React from 'react'
import { Grid } from '@material-ui/core'

import CoffeeCard from './CoffeeCard'

const Content = () => {
  return (
    <Grid container spacing={4}>
      <Grid item xs={12} sm={6} md={4}>
        <CoffeeCard 
          title={"Coffee Machine Ultra"}
          subtitle={"99.99"}
          description="The best coffee machine around! You will not regret it!"
        />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <CoffeeCard />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <CoffeeCard />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <CoffeeCard />
      </Grid>
      <Grid item xs={12} sm={6} md={4}>
        <CoffeeCard />
      </Grid>
    </Grid>

    // <CoffeeCard />
  )
}

export default Content