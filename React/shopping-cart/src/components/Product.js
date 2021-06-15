import React, { useContext } from "react";
import { GlobalContext } from '../context/GlobalState'

const Product = ({product}) => {

  const { addToCart } = useContext(GlobalContext);
  console.log("Global provider", useContext(GlobalContext));
  return (
    <div className="card h-100">
      <div className="card-body">
        <h3 className="cart-title">{ product.text }</h3>
      </div>
      <div className="card-footer d-flex justify-content-between align-items-center">
        <p className="card-text">${ product.price }</p>
        <button className="btn btn-outline-success" onClick={() => addToCart(product)}>Add to Cart</button>
      </div>
    </div>
  );
};

export default Product;
