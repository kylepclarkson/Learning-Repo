import "./App.css";

import ProductList from "./components/ProductList";
import Cart from "./components/Cart";

import { GlobalProvider } from "./context/GlobalState";

function App() {
  return (
    <GlobalProvider>
      <div className="container row">
        <ProductList />
        <Cart />
      </div>
    </GlobalProvider>
  );
}

export default App;
