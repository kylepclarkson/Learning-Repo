import './App.css';

import Greet from './components/Greet'
import Welcome from './components/Welcome'
import Message from './components/Message'
import Counter from './components/Counter'
import FunctionClick from './components/FunctionClick'
import ClassClick from './components/ClassClick'
import EventBind from './components/EventBind'
import ParentComponent from './components/ParentComponent'
import UserGreeting from './components/UserGreeting'
import NameList from './components/NameList'
import Stylesheet from './components/Stylesheet'
import Inline from './components/Inline'
import Form from './components/Form'
import FragmentDemo from './components/FragmentDemo'
import Table from './components/Table'
import RefDemo from './components/RefsDemo'
import FocusInput from './components/FocusInput'
import PortalDemo from './components/PortalDemo'
import Hero from './components/Hero'
import ErrorBoundary from './components/ErrorBoundary'
import ComponentC from './components/ComponentC'
import { UserProvider } from './components/UserContext';

import HookCounter from './hooks/HookCounter'
import HookCounterTwo from './hooks/HookCounterTwo'
import HookCounterThree from './hooks/HookCounterThree'
import HookCounterFour from './hooks/HookCounterFour'
import HookCounterOne from './hooks/HookCounterOne'
import IntervalHookCounter from './hooks/IntervalHookCounter'
import CounterOne from './hooks/CounterOne'


function App() {
  return (
    <div className="App">

      <CounterOne />

      {/* <Message /> */}
      {/* <Counter /> */}
      {/* <FunctionClick /> */}
      {/* <ClassClick /> */}
      {/* <EventBind /> */}
      {/* <ParentComponent /> */}
      {/* <UserGreeting /> */}
      {/* <NameList /> */}
      {/* <Stylesheet primary={true}/> */}
      {/* <Inline /> */}
      {/* <Form /> */}
      {/* <FragmentDemo /> */}
      {/* <Table /> */}
      {/* <RefDemo /> */}
      {/* <FocusInput /> */}
      {/* <PortalDemo /> */}
      {/* <ErrorBoundary> 
      <Hero heroName={'Justin'} />
      <Hero heroName={'Joker'} />
      </ErrorBoundary>  */}
      {/* <UserProvider value="Kyle">
        <ComponentC />
      </UserProvider> */}
      {/* <HookCounter /> */}
      {/* <HookCounterTwo /> */}
      {/* <HookCounterThree /> */}
      {/* <HookCounterFour /> */}
      {/* <HookCounterOne /> */}
      {/* <IntervalHookCounter /> */}
    </div>
  );
}

export default App;
