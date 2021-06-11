/**
 * Combines all reducers under the single root reducer. 
 */

 import { combineReducers } from 'redux'
 // Each reducer defines and evaluates the actions. 
 import postReducer from './postReducer'
 
 export default combineReducers({
     posts: postReducer
 })