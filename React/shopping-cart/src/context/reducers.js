
// === ACTIONS ===
export const ADD_PRODUCT = "ADD_PRODUCT";
export const REMOVE_PRODUCT = "REMOVE_PRODUCT";
export const CREATE_PRODUCT = "CREATE_PRODUCT";
export const DELETE_PRODUCT = "DELETE_PRODUCT";

// === Reducer functions ===
const addProductToCart = (state, product) => {
  
  const updatedState = state.map(x => x);
  const index = updatedState.findIndex((item) => item.id === product.id)
  if (index < 0) {
    updatedState.push({...product, quantity: 1})
  } else {
    const updatedItem = {...updatedState[index]};
    updatedItem.quantity++;
    updatedState[index]=updatedItem;
  }
  return updatedState
}

const removeFromCart = (state, productId) => {
  const updatedState = [...state];
  const index = updatedState.findIndex((item) => item.id === productId);
  const updatedItem = {...updatedState[index]};
  updatedItem.quantity--;
  if (updatedItem.quantity <= 0) {
    updatedState.splice(index, 1);
  } else {
    updatedState[index] = updatedItem;
  }
  return updatedState;
}

const createProduct = (state, product) => {
  console.log("Create product")
  return {...state}
}

const deleteProduct = (state, productId) => {
  console.log("Remove product")
  return {...state}
}

// === REDUCERS ===
export const cartReducer = (state, action) => {

  switch(action.type) {
    case ADD_PRODUCT:
      return addProductToCart(state, action.product);

    case REMOVE_PRODUCT:
      return removeFromCart(state, action.productId);

    default:
      return state;
  }
}

export const productReducer = (state, action) => {
  switch (action.type) {
    case CREATE_PRODUCT:
      return createProduct(state, action.product);
    case DELETE_PRODUCT:
      return deleteProduct(state, action.productId);
    default:
      return state 
  }
}

