
// === ACTIONS ===
export const ADD_PRODUCT = "ADD_PRODUCT";
export const REMOVE_PRODUCT = "REMOVE_PRODUCT";

export const FIRE_EMPLOYEE = "FIRE_EMPLOYEE";
export const MESSAGE_EMPLOYEE = "MESSAGE_EMPLOYEE";

const addProductToCart = (state, product) => {
  console.log("Adding to cart", product.id);
  console.log(state)
  return {...state}
} 

const removeProductFromCart = (state, productId) => {
  console.log("Removing from cart", productId);
  console.log(state)
  return {...state}
}

const fireEmployee = (state, employeeId) => {
  console.log("Firing Employee", employeeId);
  console.log(state)
  return {...state}
} 

export const cartReducer = (state, action) => {
  switch (action.type) {
    case ADD_PRODUCT:
      return addProductToCart(state, action.product)
    case REMOVE_PRODUCT:
      return removeProductFromCart(state, action.productId)
    default: 
      return state;
  }
}

export const employeeReducer = (state, action) => {
  switch (action.type) {
    case FIRE_EMPLOYEE:
      return fireEmployee(state, action.employeeId);
    default:
      return state;
  }
}

