import React, { useState, createContext } from 'react';

export const MovieContext = createContext();

// Provides data to components
export const MovieProvider = (props) => {
    const [movies, setMovies] = useState([
        {
            name: 'David\'s Big Day',
            price: 20,
            id: 12345
        },
        {
            name: 'Avatar',
            price: 40,
            id: 1242
        },
        {
            name: 'Rooster Attact',
            price: 5,
            id: 15
        },
    ]);
    // pass data above to components in props.children
    return (
        <MovieContext.Provider value={[movies, setMovies]}>
            {props.children}
        </MovieContext.Provider>
    );
}