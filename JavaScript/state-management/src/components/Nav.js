import './../App.css';

import React, { useContext } from 'react';
import {MovieContext} from './MovieContext';
import Movie from './Movie';

const Nav = (props) => {
    const [movies, setMovies] = useContext(MovieContext);
    return (
        <div>
            <h3>Kyle</h3>
            <p>Number of Movies {movies.length}</p>
        </div>
    );
}

export default Nav;