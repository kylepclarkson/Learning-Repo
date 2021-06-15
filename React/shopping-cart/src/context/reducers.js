// === ACTIONS ===
export const ADD_PRODUCT = "ADD_PRODUCT";
export const REMOVE_PRODUCT = "REMOVE_PRODUCT";
export const CREATE_PRODUCT = "CREATE_PRODUCT";
export const DELETE_PRODUCT = "DELETE_PRODUCT";

// === Reducer functions ===
const addProductToCart = (state, product) => {
  console.log("Add to cart called.", state.cart);
  console.log("Add to cart called.", state);
  // const index = state.cart.findIndex
  return state
}

const removeFromCart = (state, productId) => {
  console.log("Remove from cart called");
  return {...state}
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
  console.log("state", state)
  switch(action.type) {
    case ADD_PRODUCT:
      return addProductToCart(state, action.product);

    case REMOVE_PRODUCT:
      return removeFromCart(state, action.productID);

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

