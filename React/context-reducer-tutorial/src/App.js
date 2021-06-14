import "./App.css";

import GlobalState from "./context/GlobalState";

function App() {
  return (
    <GlobalState>
      <button>Add to cart</button>
      <button>Remove from cart</button>
      <button>Fire Employee</button>
    </GlobalState>
  );
}

export default App;
