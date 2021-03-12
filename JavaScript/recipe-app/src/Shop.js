import React, {useState, useEffect} from 'react';
import './App.css';


function Shop() {

    useEffect(() => {
        fetchItems();
    }, [])

    const [items, setItems] = useState([]);

    const fetchItems = async () => {
        const data = await fetch('https://fakestoreapi.com/products')
        
        const items = await data.json()
        console.log(items);
        setItems(items);
    };


  return (
      <div>
        {items.map(item => (
            <h1 key={item.id}>{item.title}</h1>
        ))}
      </div>
  );
}

export default Shop;
