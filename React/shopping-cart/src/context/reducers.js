// === ACTIONS ===
export const ADD_PRODUCT = "ADD_PRODUCT";
export const REMOVE_PRODUCT = "REMOVE_PRODUCT";

// === Reducer functions ===
const addProductToCart = (state, product) => {
  console.log("Add to cart called.")
}

const removeFromCart = (state, productId) => {
  console.log("Remove from cart called");
}

// === REDUCERS ===
export const cartReducer = (state, action) => {
  switch(action.type) {
    case ADD_PRODUCT:
      return addProductToCart(state, action.product);

    case REMOVE_PRODUCT:
      return removeFromCart(state, action.productID);

    default:
      return state;
  }
}

