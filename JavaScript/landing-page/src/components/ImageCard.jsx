import React from 'react';
import { Collapse } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardActionArea from '@material-ui/core/CardActionArea';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';

const useStyles = makeStyles({
    root: {
        maxWidth: 645,
        background: 'rgba(0,0,0,0.5)',
        margin: '40px'
    },
    media: {
        height: 440,
    },
    title: {
        fontFamily: 'Blinker',
        fontWeight: 'bold',
        fontSize: '2rem',
        color: '#fff'
    },
    description: {
        fontFamily: 'Blinker',
        fontSize: '1.25rem',
        color: '#ddd'
    }
});

export default function ImageCard({ place, checked }) {
    const classes = useStyles();

    return (
        <Collapse 
            in={checked}
            { ...(checked ? {timeout: 2000 } : {})}
            // collapsedHeight={50}
        >
            <Card className={classes.root}>
                <CardMedia
                    className={classes.media}
                    image={place.imageUrl}
                    title="Contemplative Reptile"
                />
                <CardContent>
                    <Typography
                        className={classes.title}
                        gutterBottom
                        variant="h5"
                        component="h1">
                        {place.title}
                    </Typography>
                    <Typography
                        className={classes.description}
                        variant="body2"
                        color="textSecondary"
                        component="p">
                        {place.description}
                    </Typography>
                </CardContent>
            </Card>
        </Collapse>
    );
}