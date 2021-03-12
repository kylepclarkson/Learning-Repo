/**
 * The page that shows additional detail items.
 */

import React, { useState, useEffect } from 'react';
import './App.css';
import { Link } from 'react-router-dom';

// use match to get access to parameters including item id. 
function ItemDetail({ match }) {
    useEffect(() => {
        console.log(match)
        fetchItem(match);
    }, [])

    const [item, setItem] = useState({});

    const fetchItem = async () => {
            const data = await fetch(`https://fakestoreapi.com/products/${match.params.id}`); 
            const item = await data.json()
            console.log(item)
            setItem(item)
    }

    return (
        <div>
            <h1>{item.name}</h1>
        </div>
    );
}

export default ItemDetail;

