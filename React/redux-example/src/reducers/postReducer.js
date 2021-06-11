import { FETCH_POSTS, NEW_POST } from "../actions/types";

const initialState = {
  items: [], // posts from action
  item: {}, // post we added
};

export default function (state = initialState, action) {
  // actions must have type, and will have a payload
  console.log("reducer action: ", action);
  switch (action.type) {
    case FETCH_POSTS:
      return {
        ...state,
        items: action.payload, // set posts
      };

    case NEW_POST:
      return {
        ...state,
        items: action.payload,
      };

    default:
      return state;
  }
}
