import React, { useContext } from 'react'
import { GlobalContext } from '../context/GlobalState'

import CartItem from './CartItem'

const Cart = () => {

  // Cart contents form context
  const { cart } = useContext(GlobalContext);
  console.log('cart contents', useContext(GlobalContext))
  return (
    <div className="col-md-4 col-sm-12">
      <h1>Your Shopping Cart</h1>
      <div className="row">
        {cart.map((item) => (
          <div className="col" key={item.id}>
            <CartItem item={item} />
          </div>
        ))}
      </div>
    </div>
  )
}

export default Cart
