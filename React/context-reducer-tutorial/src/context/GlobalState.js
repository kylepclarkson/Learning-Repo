import React, { createContext, useReducer } from "react";

import {
  cartReducer,
  employeeReducer,
  ADD_PRODUCT,
  REMOVE_PRODUCT,
  FIRE_EMPLOYEE,
} from "./reducers";

// === Contexts ===
const Context = createContext({
  cart: [],
  employees: [],
})

const GlobalState = ({ children }) => {

  const [cartState, cartDispatch] = useReducer(cartReducer, {cart: []});
  const [employeeState, employeeDispatch] = useReducer(employeeReducer);

  const addProductToCart = (product) => {
    cartDispatch({
      type: ADD_PRODUCT,
      product: product
    })
  }

  const removeProductFromCart = (productId) => {
    cartDispatch({
      type: REMOVE_PRODUCT,
      productId
    })
  }

  const fireEmployee = (employeeId) => {
    employeeDispatch({
      type: FIRE_EMPLOYEE,
      employeeId
    })
  }


  return (
    <Context.Provider
      value={{
        cart: cartState.cart,
        addProductToCart,
        removeProductFromCart,
        fireEmployee
      }}
    >
      {children}
    </Context.Provider>
  )
}

export default GlobalState;