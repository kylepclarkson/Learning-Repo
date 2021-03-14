import React from 'react';
import {
  Typography,
  AppBar,
  Button,
  Card,
  CardActions,
  CardContent,
  CardMedia,
  CssBaseline,
  Grid,
  Toolbar,
  Container,
} from '@material-ui/core';

import { PhotoCamera } from '@material-ui/icons';
import useStyles from './styles'

const cards = [1,2,3,4,5,6,7,8,9]

const App = () => {

    const classes = useStyles();

  return (
    <>
      <CssBaseline />
      <AppBar position="relative">
        <Toolbar>
          <PhotoCamera className={classes.Icon}/>
          <Typography variant="h6">
            Photo Album
          </Typography>
        </Toolbar>
      </AppBar>
      <main>
        <div className={classes.container}>
          <Container maxWidth="sm">
            <Typography
              variant="h2"
              align="center"
              color="textPrimary"
              gutterBottom
            >
              Photo Album
            </Typography>

            <Typography
              variant="h5"
              align="center"
              color="textSecondary"
              paragraph
            >
              Hello everyone. This is my photo album! I am starting my career, so please view my photos and tell me what you think!
            </Typography>

            <div>
              <Grid container spacing={2} justify="center">
                <Grid item>
                    <Button variant="contained" color="primary" className={classes.Button}>
                        See my photos
                    </Button>
                </Grid>
                <Grid item>
                    <Button variant="outlined" color="primary" className={classes.Button}>
                        See my photos
                    </Button>
                </Grid>
              </Grid>
            </div>
          </Container>
        </div>
        <div className={classes.cardGrid} maxWidth="md">
            <Grid container spacing={4}>
                {cards.map((card) => (
                <Grid item key={card} xs={12} sm={6} md={4}> 
                    <Card className={classes.card}>
                        <CardMedia
                            classesName={classes.cardMedia}
                            image="https://source.unsplash.com/random"
                            title="Image Title"
                        />
                        <CardContent
                            className={classes.cardContent}>
                            <Typography gutterbottom variant="h5">
                                Heading
                            </Typography>
                            <Typography gutterbottom variant="h5">
                                This is a media card. Describe the content of the card here.
                            </Typography>
                            <CardActions>
                                <Button size="small" color="primary">
                                    View
                                </Button>
                                <Button size="small" color="primary">
                                    Edit
                                </Button>
                            </CardActions>
                        </CardContent>
                    </Card>
                </Grid>
                ))}
            </Grid>

        </div>
      </main>

        <footer className="footer"> 
            <Typography variant="h6" align="center" gutterbottom>
                    Footer
            </Typography>
        </footer>
    </>
  );
}

export default App;