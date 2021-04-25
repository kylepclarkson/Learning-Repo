import React from 'react'
import { AppBar, Toolbar, Typography } from '@material-ui/core'
import { makeStyles } from '@material-ui/styles'
import AcUnitRoundedIcon from '@material-ui/icons/AcUnitRounded'

// Define css 
const useStyles = makeStyles(() => ({
  typographyStyles: {
    flex: 1
  }
}));

const Header = () => {
  // get defined css. 
  const classes = useStyles()

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography className={classes.typographyStyles}>This is the header</Typography>
        <AcUnitRoundedIcon />
      </Toolbar>
    </AppBar>
  )
}

export default Header;