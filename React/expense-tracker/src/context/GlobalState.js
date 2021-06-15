import React, { createContext, useReducer } from "react";
import AppReducer from './AppReducer';

/** Global state of the app */

const initialState = {
  transactions: [
    { id: 1, text: "Takeout", amount: -30 },
    { id: 2, text: "Job", amount: 300 },
    { id: 3, text: "Camera", amount: -100 },
    { id: 4, text: "Art", amount: -20 },
    { id: 5, text: "Store", amount: 50 },
  ],
};

// Export context so components can use it.
export const GlobalContext = createContext(initialState);

// Provider component. Will wrap the app; the children are the components with the app.
export const GlobalProvider = ({ children }) => {
  const [state, dispatch] = useReducer(AppReducer, initialState);

  // Actions for reducer
  function deleteTransaction(id) {
    dispatch({
      type: 'DELETE_TRANSACTION',
      payload: id
    })
  }

  function addTransaction(transaction) {
    dispatch({
      type: 'ADD_TRANSACTION',
      payload: transaction
    });
  }

  // Return provider. Add to value the state that is provided 
  return <GlobalContext.Provider value={{
    transactions: state.transactions,
    deleteTransaction,
    addTransaction,
    monkey: 'asdf'
  }}>{children}</GlobalContext.Provider>;
};
