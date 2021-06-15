import React, { useContext } from 'react'
import { GlobalContext } from '../context/GlobalState'

import CartItem from './CartItem'

const Cart = () => {

  // Cart contents form context
  const { cart } = useContext(GlobalContext);
  console.log('cart contents', cart)
  return (
    <div className="col-md-3 col-sm-12">
      <h3>Your Shopping Cart</h3>
        {cart.map((item) => (
          <div className="row" key={item.id}>
            <CartItem item={item} />
          </div>
        ))}
    </div>
  )
}

export default Cart
