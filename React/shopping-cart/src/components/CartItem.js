import React, { useContext } from "react";
import { GlobalContext } from "../context/GlobalState";

const CartItem = ({ item }) => {
  console.log("Item in cart", item)
  const { removeFromCart, addToCart } = useContext(GlobalContext)
  return (
    <div>
      <span>{item.text}</span> 
      <button className="btn-sm btn-outline-danger mx-2" onClick={() => removeFromCart(item.id)}>-</button>
      {item.quantity} 
      <button className="btn-sm btn-outline-success mx-2" onClick={() => addToCart(item)}>+</button>
    </div>
  );
};

export default CartItem;
