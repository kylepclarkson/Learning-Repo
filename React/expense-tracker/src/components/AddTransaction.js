import React, { useContext } from 'react'
import { GlobalContext } from '../context/GlobalState';

export const AddTransaction = () => {

  const { addTransaction } = useContext(GlobalContext);
  console.log('Add transaction', addTransaction);
  console.log('Global context', useContext(GlobalContext));

  const [text, setText] = React.useState("");
  const [amount, setAmount] = React.useState(0);

  // Add new transaction
  const onSubmit = (e) => {
    e.preventDefault();
    const newTransaction = {
      id: Math.floor(Math.random() * 1000000),
      text,
      amount: +amount // parse to number
    }
    addTransaction(newTransaction);
  }

  return (
    <div>
      <h3>Add New Transaction</h3>
      <form onSubmit={onSubmit}>
        <div className="form-control">
          <label htmlFor="text">Text</label>
          <input type="text" value={text} onChange={(e) => setText(e.target.value)} placeholder="Enter text..." />
        </div>
        <div className="form-control">
          <label htmlFor="amount">Amount <br/></label>
          <input type="number" value={amount} onChange={(e) => setAmount(e.target.value)} placeholder="Enter amount..." />
        </div>
        <button className="btn">Add transaction</button>
      </form>
    </div>
  )
}
