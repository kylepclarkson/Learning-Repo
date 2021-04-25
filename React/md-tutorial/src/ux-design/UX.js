import React from 'react'
import { Grid } from '@material-ui/core'

import Header from './components/Header'
import Content from './components/Content'

const UX = () => {
  return (
    <Grid container direction="column">
      <Grid item>
        <Header />
      </Grid>
      <Grid item container>
        <Grid item xs={0} sm={2}/>
        <Grid item xs={12} sm={8} >
          <Content />
      </Grid>
        <Grid item xs={0} sm={2}/>
      </Grid>
    </Grid>
  )
}

export default UX;