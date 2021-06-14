import React, { useReducer, createContext } from "react";

import { ADD_PRODUCT, REMOVE_PRODUCT, cartReducer } from "./reducers";

// === Context initial values ===
export const GlobalContext = createContext({
  products: [
    { id: 1, text: "Peach Pit", price: 20 },
    { id: 2, text: "Mac Demarco", price: 17 },
    { id: 3, text: "Arctic Monkeys", price: 40 },
    { id: 4, text: "The Rolling Stones", price: 18 },
    { id: 5, text: "Motorhead", price: 23 },
    { id: 6, text: "Monte", price: 15 },
  ],
  cart: [],
});

// === Provider of context ===
export const GlobalProvider = (props) => {
  // cart state.
  const [cartState, cartDispatch] = useReducer(cartReducer);

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

  return (
    <GlobalContext.Provider
      value={{
        cart: cartState.cart,
        addToCart: addToCart,
        removeFromCart: removeFromCart,
      }}
    >
      {props.children}
    </GlobalContext.Provider>
  );
};
