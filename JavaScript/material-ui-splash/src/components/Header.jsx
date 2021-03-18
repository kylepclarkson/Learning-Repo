import React from 'react'
import {AppBar, Toolbar, Typography } from '@material-ui/core'
import AcUnitRoundedIcon from '@material-ui/icons/AcUnitRounded'
import { makeStyles } from '@material-ui/styles' 

const useStyles = makeStyles(() => ({
    toolbar: {
        boxShadow: "none",
        background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
    },
    typographyStyles: {
        flex: 1,
    },
}));

function Header() {
    const classes = useStyles();
    return (
        <AppBar position='static'>
            <Toolbar className={classes.toolbar}>
                <Typography className={classes.typographyStyles}>
                    Spciy Meatballs
                </Typography>
                <AcUnitRoundedIcon />
            </Toolbar>
        </AppBar>
    );
};

export default Header
