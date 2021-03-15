
import './App.css';
import Navbar from './components/Navbar'
import Home from './components/pages/Home'
import {BrowserRouter as Router, Switch, Route} from 'react-router-dom'

function App() {
  return (
    <>
      <Router>
        <Navbar />
        <Switch>
          <Route path='/' exact component={Home} /> {/** Home */}
        </Switch>
      </Router>
    </>
  );
}

export default App;
