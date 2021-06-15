import React, { useReducer, createContext } from "react";

import { ADD_PRODUCT, REMOVE_PRODUCT, CREATE_PRODUCT, DELETE_PRODUCT, cartReducer, 
  productReducer  } from "./reducers";

// === Context initial values ===
export const GlobalContext = createContext({
  // products: [
  //   { id: 1, text: "Peach Pit", price: 20 },
  //   { id: 2, text: "Mac Demarco", price: 17 },
  //   { id: 3, text: "Arctic Monkeys", price: 40 },
  //   { id: 4, text: "The Rolling Stones", price: 18 },
  //   { id: 5, text: "Motorhead", price: 23 },
  //   { id: 6, text: "Monte", price: 15 },
  // ],
  // cart: [],
});

// === Provider of context ===
export const GlobalProvider = (props) => {
  // cart state.
  const [cartState, cartDispatch] = useReducer(cartReducer, [
    { id: 1, text: 'Peach Pit', price: 20, quantity: 2}
  ]);
  const [productState, productDispatch] = useReducer(productReducer, [
    { id: 1, text: "Peach Pit", price: 20 },
    { id: 2, text: "Mac Demarco", price: 17 },
    { id: 3, text: "Arctic Monkeys", price: 40 },
    { id: 4, text: "The Rolling Stones", price: 18 },
    { id: 5, text: "Motorhead", price: 23 },
    { id: 6, text: "Monte", price: 15 },
  ]);

  // === DISPATCHES ===
  const addToCart = (product) => {
    cartDispatch({
      type: ADD_PRODUCT,
      product: product,
    });
  };

  const removeFromCart = (productId) => {
    cartDispatch({
      type: REMOVE_PRODUCT,
      productId: productId,
    });
  };

  const createProduct = (product) => {
    productDispatch({
      type: CREATE_PRODUCT,
      product: product
    })
  }

  const deleteProduct = (productId) => {
    productDispatch({
      type: DELETE_PRODUCT,
      productId
    })
  }

  return (
    <GlobalContext.Provider
      value={{
        cart: cartState,  // Issue: Using multiple states instead of a global state.
        products: productState,
        addToCart,
        removeFromCart,
        createProduct,
        deleteProduct
      }}
    >
      {props.children}
    </GlobalContext.Provider>
  );
};
