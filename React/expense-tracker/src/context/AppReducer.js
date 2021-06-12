// Reducer to change state 

export default (state, action) => {
  switch(action.type) {
    case 'DELETE_TRANSACTION':
      // Create new state, filtering out deleted state. .
      return {
        ...state,
        transactions: state.transactions.filter(transaction => transaction.id !== action.payload)
      }

    case 'ADD_TRANSACTION':
      // Add new transaction to transactions.
      return {
        ...state,
        transactions: [action.payload, ...state.transactions]
      }

    default:
      return state;
  }
}